import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as apigw from "aws-cdk-lib/aws-apigatewayv2";
import * as integrations from "aws-cdk-lib/aws-apigatewayv2-integrations";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";

export class MlopsInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Get build number from context
    const buildNumber = this.node.tryGetContext("buildNumber") || "latest";

    // Look up the default VPC
    const vpc = ec2.Vpc.fromLookup(this, "DefaultVpc", {
      isDefault: true,
    });

    // S3 bucket for data
    const dataBucket = new s3.Bucket(this, "DataBucket", {
      bucketName: "data-bucket-resche",
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // S3 bucket for models
    const modelBucket = new s3.Bucket(this, "ModelBucket", {
      bucketName: "model-bucket-resche",
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // ECS Cluster using the default VPC
    const cluster = new ecs.Cluster(this, "EcsCluster", {
      vpc: vpc,
    });

    // ECR Repository for the training Docker image
    const repository = new ecr.Repository(this, "MLECRRepo", {
      repositoryName: "ml-ecr-repo",
    });

    // ECS Task Definition for Fargate
    const taskDef = new ecs.FargateTaskDefinition(this, "TaskDef", {
      cpu: 1024,
      memoryLimitMiB: 2048,
    });

    // Add container with specific build number tag
    const container = taskDef.addContainer("Cifar10TrainingContainer", {
      image: ecs.ContainerImage.fromEcrRepository(repository, buildNumber),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "Cifar10Training" }),
      environment: {
        DATA_BUCKET_NAME: dataBucket.bucketName,
        MODEL_BUCKET_NAME: modelBucket.bucketName,
        S3_OBJECT_KEY: "cifar-10-batches-py/",
        MODEL_NAME: `model-${buildNumber}.pth`,
      },
    });

    container.addPortMappings({ containerPort: 80 });

    // IAM permissions for ECS task role to access S3 buckets
    taskDef.addToTaskRolePolicy(
      new iam.PolicyStatement({
        actions: [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:GetAuthorizationToken",
        ],
        resources: [
          `${dataBucket.bucketArn}/*`,
          `${modelBucket.bucketArn}/*`,
          repository.repositoryArn,
        ],
      })
    );

    // Lambda Function for inference
    const inferenceLambda = new lambda.Function(this, "InferenceLambda", {
      runtime: lambda.Runtime.FROM_IMAGE,
      code: lambda.Code.fromEcrImage(repository, { tag: "lambda" }),
      handler: lambda.Handler.FROM_IMAGE,
      environment: {
        MODEL_BUCKET_NAME: modelBucket.bucketName,
      },
      memorySize: 3000,
      timeout: cdk.Duration.seconds(61),
    });

    // IAM Permissions for Lambda to access S3 bucket
    inferenceLambda.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket", "s3:GetObject"],
        resources: [modelBucket.bucketArn, `${modelBucket.bucketArn}/*`],
      })
    );

    // **API Gateway for Lambda**
    const httpApi = new apigw.HttpApi(this, "HttpApi", {
      apiName: "MLOpsInferenceAPI",
      description: "HTTP API for image inference requests",
      corsPreflight: {
        allowOrigins: ["*"], 
        allowMethods: [apigw.CorsHttpMethod.POST], 
        allowHeaders: ["Content-Type", "Authorization"],
        maxAge: cdk.Duration.days(10),
      },
    });

    // Create Lambda Integration
    const lambdaIntegration = new integrations.HttpLambdaIntegration(
      "InferenceLambdaIntegration",
      inferenceLambda
    );

    // Create API Route for inference
    httpApi.addRoutes({
      path: "/predict",
      methods: [apigw.HttpMethod.POST],
      integration: lambdaIntegration,
    });

    // Output the API URL
    new cdk.CfnOutput(this, "APIEndpoint", {
      value: httpApi.apiEndpoint,
      description: "HTTP API Endpoint for inference",
    });

    // **Frontend Hosting: S3 Bucket and CloudFront**
    const frontendBucket = new s3.Bucket(this, "FrontendBucket", {
      websiteIndexDocument: "index.html",
      websiteErrorDocument: "index.html",
      publicReadAccess: false, 
      removalPolicy: cdk.RemovalPolicy.DESTROY, 
      autoDeleteObjects: true,
      blockPublicAccess: {
        blockPublicAcls: false,
        blockPublicPolicy: false,
        ignorePublicAcls: false,
        restrictPublicBuckets: false,
      }
    });

    // CloudFront Origin Access Identity (OAI)
    const cloudfrontOAI = new cloudfront.OriginAccessIdentity(this, "OAI", {
      comment: "OAI for FrontendBucket",
    });

    // Grant CloudFront access to S3 bucket
    frontendBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject"],
        resources: [frontendBucket.arnForObjects("*")],
        principals: [new iam.CanonicalUserPrincipal(cloudfrontOAI.cloudFrontOriginAccessIdentityS3CanonicalUserId)],
      })
    );

    // CloudFront distribution for the frontend
    const distribution = new cloudfront.CloudFrontWebDistribution(this, "FrontendDistribution", {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: frontendBucket,
            originAccessIdentity: cloudfrontOAI,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
              allowedMethods: cloudfront.CloudFrontAllowedMethods.GET_HEAD,
              compress: true,
            },
          ],
        },
      ],
      defaultRootObject: "index.html",
    });

    // Output the CloudFront Distribution URL
    new cdk.CfnOutput(this, "CloudFrontURL", {
      value: `https://${distribution.distributionDomainName}`,
      description: "CloudFront Distribution URL for your frontend",
    });
  }
}

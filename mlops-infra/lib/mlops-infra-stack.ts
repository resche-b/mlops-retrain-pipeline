import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3_notifications from 'aws-cdk-lib/aws-s3-notifications';
import * as iam from 'aws-cdk-lib/aws-iam';

export class MlopsInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

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
    const repository = new ecr.Repository(this, 'EcrRepository');

    // Define ECS Task Definition for Fargate service
    const taskDef = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      cpu: 1024,  // 1 vCPU
      memoryLimitMiB: 2048,  // 2 GB of memory
    });

    const container = taskDef.addContainer('Cifar10TrainingContainer', {
      image: ecs.ContainerImage.fromEcrRepository(repository),
      gpuCount: 0,
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'Cifar10Training' }),
      environment: {
        // Pass environment variables to the container
        DATA_BUCKET_NAME: dataBucket.bucketName,
        MODEL_BUCKET_NAME: modelBucket.bucketName,
        S3_OBJECT_KEY: "cifar-10-batches-py/", 
      },
    });

    container.addPortMappings({ containerPort: 80 });

    // Lambda function to trigger ECS tasks
    const triggerTrainingLambda = new lambda.Function(this, "RetrainLambda", {
      runtime: lambda.Runtime.NODEJS_16_X,
      handler: "index.handler",
      code: lambda.Code.fromAsset("lambda"),
      environment: {
        DATA_BUCKET_NAME: dataBucket.bucketName,
        CLUSTER_NAME: cluster.clusterName,
        TASK_DEF_ARN: taskDef.taskDefinitionArn,
      },
    });

    // Grant Lambda permissions to read from the data bucket
    dataBucket.grantRead(triggerTrainingLambda);

    // IAM permissions for Lambda to trigger ECS tasks and access S3
    triggerTrainingLambda.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          'ecs:RunTask',
          'ecs:DescribeTasks',
          'ecr:GetAuthorizationToken',
          'ecr:BatchGetImage',
          'ecr:GetDownloadUrlForLayer',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
        ],
        resources: ['*'], 
      })
    );

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
        resources: [`${dataBucket.bucketArn}/*`, `${modelBucket.bucketArn}/*`, repository.repositoryArn],
      })
    );
  }
}

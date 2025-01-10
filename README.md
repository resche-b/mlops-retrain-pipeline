# 🚀 MLOps Project: Automated Model Training Pipeline

This project demonstrates an end-to-end MLOps workflow, streamlining and automating model training for machine learning models using **AWS services** and **GitHub Actions**. The project includes a robust infrastructure setup, an automated pipeline for building and deploying Dockerized machine learning applications, and a **React-based frontend** for users to interact with the trained model.

## ✨ Features

- 🔄 **Automated CI/CD Pipeline**: Leveraging GitHub Actions to build and push Docker images to Amazon Elastic Container Registry (ECR), deploy infrastructure using AWS CDK, and trigger an **ECS Fargate** task that trains the model.
- 🧠 **Model Training**: Uses **Amazon ECS Fargate** to trigger training jobs on the CIFAR dataset.
- 🏗️ **Infrastructure as Code (IaC)**: All resources are provisioned using AWS CDK written in TypeScript.
- 🐳 **Containerized Workflow**: Dockerized ML application for portability and scalability.
- 🌐 **User-Friendly Frontend**: A React-based web app hosted on **Amazon S3** and distributed via **CloudFront** to provide low-latency access to users.
- ☁️ **S3 Integration**: Stores models for future inference tasks.
- ⚙️ **Efficient Resource Management**: Built on AWS's serverless and scalable services to optimize cost and performance.

## 🌍 Live Deployment

Check out the live application here: [Live Demo](https://d3bnlqzkqhdpm2.cloudfront.net/)

## 🛠️ Architecture Overview

1. **GitHub Actions**: Automates the CI/CD workflow.
   - 🛠️ Builds and pushes Docker images to **Amazon ECR**.
   - 🚀 Deploys infrastructure using AWS CDK.
   - ⚙️ Triggers an ECS Fargate task for model training.
2. **AWS ECS Fargate**: Runs training jobs in a serverless container environment.
3. **Amazon S3**: Stores the trained models for further usage and hosts the frontend assets.
4. **CloudFront**: Distributes the React web app globally for faster access.
5. **React Web App**: Provides an interface for users to interact with the trained model.

## 🧰 Technologies Used

- 🖥️ **AWS Services**: ECS Fargate, S3, ECR, CloudFront, CDK
- 📝 **Programming Languages**: TypeScript (AWS CDK), Python (ML Model), JavaScript (React Frontend)
- ⚙️ **CI/CD**: GitHub Actions
- 📊 **Dataset**: CIFAR


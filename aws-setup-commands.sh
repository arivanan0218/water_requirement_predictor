# AWS ECS Deployment Setup Script
# Run these commands in AWS CLI after setting up your credentials

# 1. Create ECR Repository
aws ecr create-repository --repository-name water-requirement-predictor --region ap-south-1

# 2. Create CloudWatch Log Group
aws logs create-log-group --log-group-name /ecs/water-requirement-task --region ap-south-1

# 3. Create ECS Cluster
aws ecs create-cluster --cluster-name water-requirement-cluster --region ap-south-1

# 4. Register Task Definition (run this after creating the IAM roles)
aws ecs register-task-definition --cli-input-json file://task-definition.json --region ap-south-1

# 5. Create ECS Service (run this after registering task definition)
aws ecs create-service \
    --cluster water-requirement-cluster \
    --service-name water-requirement-service \
    --task-definition water-requirement-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
    --region ap-south-1

# Note: Replace subnet-12345 and sg-12345 with your actual subnet and security group IDs

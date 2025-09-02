from __future__ import annotations
import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class BotoS3:
    def __init__(self, aws_profile: str | None, aws_region: str, action: str | None = None):
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self.action = action
        self.s3_client = self.get_boto_s3_client()
        self.s3_resource = self.get_boto_s3_resource()

    def get_boto_session(self):
        # IRSA (EKS)
        if os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") and os.getenv("AWS_ROLE_ARN"):
            return boto3.Session(region_name=self.aws_region)
        # Named profile (local/dev)
        if self.aws_profile:
            return boto3.Session(profile_name=self.aws_profile, region_name=self.aws_region)
        # Default chain
        return boto3.Session(region_name=self.aws_region)

    def get_boto_s3_client(self):
        return self.get_boto_session().client("s3")

    def get_boto_s3_resource(self):
        return self.get_boto_session().resource("s3")

    def bucket_exists(self, bucket_name: str) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except Exception:
            return False

    def create_s3_bucket(self, bucket_name: str):
        try:
            if self.bucket_exists(bucket_name):
                logger.info(f"S3 bucket already exists: {bucket_name}")
                return

            if self.aws_region == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.aws_region},
                )
            logger.info(f"Created S3 bucket: {bucket_name}")

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "BucketAlreadyOwnedByYou":
                logger.info(f"S3 bucket already owned by you: {bucket_name}")
            else:
                logger.error(f"Error creating bucket {bucket_name}: {e}")
                raise

    def update_bucket_policies(self, bucket_name: str):
        try:
            self.s3_client.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": False,
                    "IgnorePublicAcls": False,
                    "BlockPublicPolicy": False,
                    "RestrictPublicBuckets": False,
                },
            )
            self.s3_client.put_bucket_ownership_controls(
                Bucket=bucket_name,
                OwnershipControls={"Rules": [{"ObjectOwnership": "BucketOwnerPreferred"}]},
            )
            logger.info(f"Updated ownership/public access settings for: {bucket_name}")
        except ClientError as e:
            logger.error(f"Error updating policies for {bucket_name}: {e}")
            # Non-fatal; choose whether to re-raise

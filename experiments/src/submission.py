import os
import boto3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


# load envs from env file
load_dotenv(verbose=True, dotenv_path="/work/sigir-ecom-2021/upload.env")

# env info should be in your env file
BUCKET_NAME = os.getenv("BUCKET_NAME") # you received it in your e-mail
EMAIL = os.getenv("EMAIL") # the e-mail you used to sign up
PARTICIPANT_ID = os.getenv("PARTICIPANT_ID") # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY") # you received it in your e-mail
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY") # you received it in your e-mail


def upload_submission(local_file: str, task: str):
    print("Starting submission at {}...\n".format(datetime.utcnow()))
    # instantiate boto3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY ,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name="us-west-2",
    )
    s3_file_name = os.path.basename(local_file)
    s3_file_path = "{}/{}/{}".format(task, PARTICIPANT_ID, s3_file_name)  # it needs to be like e.g. "rec/id/*.json"
    s3_client.upload_file(local_file, BUCKET_NAME, s3_file_path)
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))


def submission(outfile_path: Path, task: str) -> None:
    replaced_email = EMAIL.replace("@", "_")
    current_datetime_ms = int(datetime.utcnow().timestamp() * 1000)
    parent = outfile_path.parent
    submit_file_path = parent / f"{replaced_email}_{current_datetime_ms}.json"
    outfile_path.rename(submit_file_path)
    upload_submission(local_file=str(submit_file_path), task=task)

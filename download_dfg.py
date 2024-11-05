import os
import tarfile
from os import path
from zipfile import ZipFile

import boto3
import yaml

if path.exists("./data/dfg/JPEGImages"):
    print("files already downloaded")
    exit()

if not path.exists("./data/dfg"):
    os.makedirs("./data/dfg")

with open("./s3.yaml", "r") as f:
    config = yaml.safe_load(f)
session = boto3.session.Session()
client = session.client("s3", **config["s3"])

client.download_file(
    "dfg", "JPEGImages.tar.bz2", path.join("./data/dfg", "JPEGImages.tar.bz2")
)
client.download_file("dfg", "dfg-annot.zip", path.join("./data/dfg", "dfg-annot.zip"))

with tarfile.open("./data/dfg/JPEGImages.tar.bz2", "r:bz2") as tar:
    tar.extractall("./data/dfg")


with ZipFile(f"./data/dfg/dfg-annot.zip", "r") as annot_zip:
    annot_zip.extractall(f"./data/dfg")

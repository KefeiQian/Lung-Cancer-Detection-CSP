
from __future__ import print_function

import shutil
import sys
from datetime import datetime

import oss2


def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')
        sys.stdout.flush()


if __name__ == '__main__':
    print("making output zip file....")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "output-{}".format(current_time)
    shutil.make_archive(filename, 'zip', "../output")

    print("oss auth....")
    auth = oss2.Auth('', '') # fill the auth token
    bucket = oss2.Bucket(auth, 'https://oss-cn-hangzhou.aliyuncs.com', '') # fill the bucket name

    print("uploading...")

    zip_file = "{}.zip".format(filename)
    with open(zip_file, 'rb') as fileobj:
        bucket.put_object(zip_file, fileobj, progress_callback=percentage)
        print("upload finished")

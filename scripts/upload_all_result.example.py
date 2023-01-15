
from __future__ import print_function

import os
import sys
from datetime import datetime
import zipfile
import oss2


def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')
        sys.stdout.flush()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder = "../output"

    print("making all zip file....")

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    zip_file = "output-{}.zip".format(current_time)

    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, folder))

    print("oss auth....")
    auth = oss2.Auth('', '')
    bucket = oss2.Bucket(auth, 'https://oss-cn-hangzhou.aliyuncs.com', '')

    print("uploading...")

    with open(zip_file, 'rb') as fileobj:
        bucket.put_object(zip_file, fileobj, progress_callback=percentage)
        print("upload finished")

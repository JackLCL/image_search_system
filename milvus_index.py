import getopt
import sys
from milvus import *

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = '19530'

def connect_server():
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        return milvus
    except Exception as e:
        print(e)

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "t",
            ["help", "table="],
        )
    except getopt.GetoptError:
        print("Usage: test.py -t <collection_name>")
        sys.exit(2)
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name == "--table":
            collection_name = opt_value

    index_type = IndexType.IVF_FLAT
    index_param = {'nlist': 16384}
    print(collection_name, " ", index_type, " ", index_param)

    milvus = connect_server()
    status = milvus.create_index(collection_name,index_type,index_param)
    print(status)


if __name__ == '__main__':
    main()
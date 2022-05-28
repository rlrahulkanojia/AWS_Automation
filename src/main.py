import os
import argparse
MODE = 'V1'
import V1
import V2 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    value = parser.parse_args()

    if MODE == 'V1':
        V1.V1_analytics(value.name)
    
    if MODE == 'V2':
        V2.V2_analytics(value.name)
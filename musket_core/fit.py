import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Analize experiment metrics.')
    parser.add_argument('--inputFolder', type=str, default=".",
                        help='folder to search for experiments')
    
    pass

if __name__ == '__main__':
    main()
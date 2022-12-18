import os

def getPath(): 
    path = os.path.abspath(__file__).strip(os.path.basename(__file__))
    return path
if __name__ == "__main__":
    print (getPath())
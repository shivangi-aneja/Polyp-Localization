import cv2
import os

width = 512
height = 512
dim = (width, height)
yourpath = os.getcwd()
for root, dirs, files in os.walk(yourpath, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".PNG"):
                print ("A PNG file already exists for %s" % name)
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".PNG"
                try:
                    img = cv2.imread(os.path.join(root, name))
                    print('Original Dimensions : ', img.shape)
                    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                    print('Resized Dimensions : ', resized.shape)
                    cv2.imwrite(outfile, resized)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                except Exception:
                    print("error")

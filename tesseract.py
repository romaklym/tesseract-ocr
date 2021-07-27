import cv2
import pytesseract
import numpy as np
from textblob import TextBlob
import json
import os

img = cv2.imread('D:\\find_text.png')

final_height, final_width = img.shape[:2]
image = cv2.resize(img, (int(final_width//2),
                   int(final_height//2)), interpolation=cv2.INTER_CUBIC)


cv2.imshow('BEFORE', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def resized_image(img, scale_x, scale_y):
    if scale_x == 1 and scale_y == 1:
        return img

    original_height, original_width = img.shape[:2]
    img = cv2.resize(img, (int(original_width * scale_x),
                     int(original_height * scale_y)), interpolation=cv2.INTER_CUBIC)
    return img


def load_config_file(config):

    if config is None:
        return False
    if not os.path.exists(config):
        return False
    try:
        with open(config, 'r', encoding="utf8") as json_file:
            config = json.load(json_file)
    except Exception as exc:
        print(repr(exc))

        return False

    return config

# config = load_config_file("D:\\CODE\\mad_ui_test_automation\\mad\\tests\\LanguageConfigs\\debug_language_config.json")
# config['japanese_commands'][0]['avatar']
# print(config['japanese_commands'][0]['avatar'])


#img = resized_image(img, scale_x=2, scale_y=2)
#ret, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.bitwise_not(img)
#kernel = np.ones((3,3), np.uint8)

#img = cv2.GaussianBlur(img, (5, 5), 0)
#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# FOR BRIGHT IMAGES
#ret, img = cv2.threshold(img, 205, 255, cv2.THRESH_BINARY_INV)
# FOR DARK IMAGES
#ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)
#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
final_height, final_width = img.shape[:2]
image = cv2.resize(img, (int(final_width//2),
                   int(final_height//2)), interpolation=cv2.INTER_CUBIC)

cv2.imshow('AFTER', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

custom_config = r"--psm 6 -c tessedit_char_whitelist=0123456789."
#  -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!()[]_-""0123456789
language = 'eng'
#image_array = Image.fromarray(img)
text = pytesseract.image_to_string(img, lang=language, config=custom_config)

file = open("D:\\result.txt", "w", encoding="utf8")
file.write(text)


# if config['japanese_commands'][0]['avatar'] in text:
#     print("bingo")

text = text.replace('\f', '')


def save_image(name, image, path):
    path_for_image = '{}\{}.png'.format(path, name)
    cv2.imwrite(path_for_image, image)


save_image('final_result', img, "D:")

# translate the text into a different language
if 'eng' in language:
    if "10.41.84.124" in text:
        print(text)
        print("YES")
    else:
        print(text)
else:
    tb = TextBlob(text)
    translated = tb.translate(to="en")
    # show the translated text
    print("TRANSLATED TO:")
    print(translated)
    print("OG:")
    print(text)

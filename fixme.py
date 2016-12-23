import io

notebook = 'Research_Update.ipynb'
path = notebook[:-6] + '.slides.html'
flag = u'@media print{*{text-shadow:none !important;color:#000 !important'

with io.open(path, 'r') as in_file:
    data = in_file.readlines()
    for i, line in enumerate(data):
        if line[:64] == flag:
            data[i] = data[i].replace('color:#000 !important;', '')

with io.open(path, 'w') as out_file:
    out_file.writelines(data)

print("You can now print your slides")

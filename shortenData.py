import csv

f = open("song_lyrics.csv", 'r', newline='')
read = csv.reader(f)
rows = list(read)
f.close()

rows = rows[:15000]

file = open("newsong.csv", 'w', newline='')
write = csv.writer(file)
write.writerows(rows)
file.close()
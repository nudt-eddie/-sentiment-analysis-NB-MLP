import csv

file = 'douyin_bian.sql'
sql = open(file, 'r', encoding='UTF-8')
head_pos = 0
headers = ['id', 'ppc', 'note_id', 'title', 'description', 'liked_count', 'nickname', 'userid', 'pub_date', 'spider_date', 'biandao_score']
with open('data.csv', "w", encoding="utf-8", newline="") as w:
    writer_1 = csv.writer(w)
    writer_1.writerow(headers)
    for data in sql:
        if data[0:6] == 'INSERT':
            head_pos = data.index('(')
            use_data = data[head_pos + 1:-3].split('),(')
            for i in use_data:
                if len(i.split(',')) != 11:
                    continue
                writer_1.writerow(i.replace("'",'').split(','))
sql.close()

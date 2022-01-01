from random import randint, uniform
aracyasi = []
kilometre = []
yakittuketim = []
motorbeygir = []
finale = []
for i in range(100):
    new_data = []
    yas = randint(1, 15)
    km = randint(250, 10000) * yas
    motor = randint(100,250)
    motor = motor - (motor%10) #siyah gri beyaz mavi kırmızı
    yakit = uniform(7, 15)
    yakit = yakit - (yakit%0.1)
    fiyat = int(randint(500000, 5000000) / (yas * 0.01) / (km * 0.1) * (motor * 0.5))

    new_data.append(yas)
    new_data.append(km)
    new_data.append(yakit)
    new_data.append(motor)
    new_data.append(fiyat)
    finale.append(new_data)

file = open("random.csv","w+")
data_num = 100
for data in finale:
    query = ""
    for i in range(5):
        query += str(data[i])
        if i == 5 - 1:
            break
        else:
            query += ","
    file.write(query + "\n")

file.close()

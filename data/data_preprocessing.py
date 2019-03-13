import pandas as pd
import re
from stop_words import get_stop_words
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

train_data = pd.read_csv('data/train.csv')
train_title = train_data['title'].tolist()
test_data = pd.read_csv('data/test.csv')
test_title = test_data['title'].tolist()

# incomplete
irrelevant_words = ["wa", "murah", "termurah", "original", "satuan", "ready" ,"stok", "stock", "limited", "edition", "promo", "bulan", "ini", "hari", "minggu", "gratis", "ongkir", "garansi", "pcs", "pc", "ori", "berkualitas", "new", "tanpa", "beli", "terlaris", "terbaru", "baru", "gudang", "bayar", "free", "harga", "chat", "seller", "terbatas", "premium", "jual", "paket", "minat", "khusus", "exclusive", "eksklusif", "asli", "barang", "terbaik", "laris", "distributor", "product", "bonus", "bagus", "dijual", "terjamin", "whatsapp", "serius", "real", "like", "price", "terkini", "order", "pengiriman", "info", "silahkan", "silakan", "setengah", "via", "impor", "import", "diskon", "discount", "lengkap", "terupdate", "bigsale", "obral", "dijamin", "terbukti", "discon", "dapat", "tidak", "terakhir", "buruan", "pemesanan", "pesanan", "pesan", "buy", "banget", "pilihan", "serba", "non", "ecer", "oke", "ok", "amazing", "banyak", "dus", "wholesale", "update", "updet", "bukan", "from", "tutup", "orginal", "siap", "arrival", "pre", "kirim", "habisin", "extreme", "hadiah", "selling", "kualitasnya", "boskuu", "dicari", "better", "exlusive", "palsu", "kak", "deal", "buat", "dapatkan", "tersedia", "aja", "sebelum", "terpercaya", "pasti", "langsung", "12.12", "edisi", "no.1", "seluruh", "hello", "gila", "gilaa", "variasi", "jaminan", "dapetin", "wow", "cocok", "istimewa", "recommended", "recomended", "menyambut", "masyarakat", "restock", "tren", "datang", "now", "terlengkap", "produck", "mantap", "kembali", "terjangkau", "nego", "hubungi", "berubah", "terkeren", "seru", "terbuka", "terpopuler", "sold", "cuma", "unggulan", "satuan", "very", "fake", "mesin", "geratis", "besar2an", "ekslusif", "mudah", "koleksi", "gojek", "salee", "pasang", "gransi", "packing", "pack", "label", "miliki", "imitasi", "pilih", "chating", "grtis", "suplier", "supplier", "semua", "indah", "ended", "bagian", "komplit", "klasik", "happy", "ajaib", "jamin", "nasional", "cheap", "hub", "cheapest", "eceran"]
# irrelevant_words = []
irrelevant_words.extend(get_stop_words('id'))

def delete_words(matchobj):
    word = matchobj.group(0)
    if word in irrelevant_words:
        return ""
    else:
        return word

def preprocessing(title_array):
    new_title_array = []
    units = ['ml', 'gr', 'gram', 'gb', 'mg', 'g', 'oz', 'cc', 'inch', 'mp', 'm', 'cm', 'kg']
    # temporary pattern
    units_pattern = re.compile('((\A| )\d+) +((ml)|(gr)|(gram)|(gb)|(mg)|(g)|(oz)|(cc)|(inch)|(mp)|(m)|(cm)|(kg)( |\Z))')
    for title in title_array:
        # lowercase and remove special characters
        new_title = re.sub('[^a-z0-9 ]', ' ', title.lower())
        # remove numbers
        new_title = re.sub('\d', ' ', new_title)
        # # remove irrelevant words (incomplete list)
        # new_title = re.sub('[a-z]+', delete_words, new_title)
        # # combine numbers and units (e.g. 3 ml -> 3ml)
        # new_title = units_pattern.sub(r'\1\3', new_title)
        # # remove 1-letter words
        # new_title = re.sub('(\A| )[a-z]{1}( |\Z)', ' ', new_title)
        # # remove extra space character
        # new_title = re.sub(' +', ' ', new_title)
        # # strip trailing space characters
        # new_title = new_title.strip()
        new_title = stemmer.stem(new_title)
        if new_title == '':
            new_title_array.append(' ')
        else:
            new_title_array.append(new_title)
    return new_title_array

new_train_title = preprocessing(train_title)
new_test_title = preprocessing(test_title)

train_data['title'] = new_train_title
test_data['title'] = new_test_title
train_data.to_csv(index=False, path_or_buf='data/new_train.csv')
test_data.to_csv(index=False, path_or_buf='data/new_test.csv')

print("\nData has been cleaned")
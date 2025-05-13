import re
from pyvi import ViTokenizer
import string
import pandas as pd

# Đường dẫn tới file CSV gốc
input_file = 'comments_output.csv'

# Hàm loại bỏ các ký tự đặc biệt trong chuỗi văn bản
def remove_special_text(text):
    tokens = re.sub(r'(http\S+)|(@\S+)|RT|\#|\?|:|\.|[0-9]|[!@#$%^&*()-+<>{}]|,', ' ', text)
    return tokens

# Hàm xử lý các từ viết tắt 1 ký tự
def acronym_text_one_word(text):
    text = text.replace("r", "rồi")
    text = text.replace("a", "anh")
    text = text.replace("e", "em")
    text = text.replace("j", "gì")
    text = text.replace("k", "không")
    text = text.replace("m", "mình")
    text = text.replace("t", "tôi")
    text = text.replace("b", "bạn")
    text = text.replace("h", "giờ")
    text = text.replace("s", "sao")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\t", "")
    return text

# Hàm xử lý các từ viết tắt 2 ký tự
def acronym_text_two_words(text):
    text = text.replace("ko", "không")
    text = text.replace("k0", "không")
    text = text.replace("bt", "bình thường")
    text = text.replace("vn", "việt nam")
    text = text.replace("vs", "và")
    text = text.replace("cx", "cũng được")
    text = text.replace("đc", "được")
    text = text.replace("dc", "được")
    text = text.replace("nh", "nhưng")
    text = text.replace("đg", "đường")
    text = text.replace("nc", "nước")
    text = text.replace("ms", "mới")
    text = text.replace("bh", "bao giờ")
    text = text.replace("km", "khuyến mãi")
    text = text.replace("ae", "anh em")
    text = text.replace("sg", "sài gòn")
    text = text.replace("hn", "hà nội")
    text = text.replace("vk", "vợ")
    text = text.replace("ck", "chồng")
    text = text.replace("nv", "nhân viên")
    text = text.replace("mn", "mọi người")
    text = text.replace("qc", "quảng cáo")
    text = text.replace("sp", "sản phẩm")
    text = text.replace("sd", "sử dụng")
    text = text.replace("nt", "nhắn tin")
    text = text.replace("wa", "quá")
    return text

# Hàm xử lý các từ viết tắt 3 ký tự
def acronym_text_three_words(text):
    text = text.replace("ntn", "như thế nào")
    text = text.replace("lun", "luôn")
    text = text.replace("mun", "mụn")
    text = text.replace("trc", "trước")
    text = text.replace("chx", "chưa")
    text = text.replace("nhg", "nhưng")
    text = text.replace("spt", "sản phẩm tốt")
    text = text.replace("nch", "nói chung")
    text = text.replace("kbh", "không bao giờ")
    return text

# Hàm xử lý các từ viết tắt 4 ký tự
def acronym_text_four_words(text):
    text = text.replace("tnao", "thế nào")
    text = text.replace("hqua", "hôm qua")
    text = text.replace("toẹt", "tuyệt")
    text = text.replace("nhưg", "nhưng")
    text = text.replace("hnay", "hôm nay")
    text = text.replace("thik", "thích")
    return text

# Hàm xử lý các từ viết tắt dài hơn
def acronym_text(text):
    text = text.replace("lquan", "liên quan")
    text = text.replace("Nchung", "nói chung")
    text = text.replace("k_thể", "không thể")

    return text

# Hàm chuẩn hóa văn bản
def normalize_text(text):
    # Loại bỏ các ký tự kéo dài (ví dụ: đẹppppppp)
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # Chuyển thành chữ thường
    text = text.lower()

    # Chuẩn hóa tiếng Việt, xử lý emoji, chuẩn hóa tiếng Anh và thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        # Quy các icon về 2 loại emoji: Tích cực hoặc tiêu cực
        "👹": "nagative", "👻": "positive", "💃": "positive",'🤙': ' positive ', '👍': ' positive ',
        "💄": "positive", "💎": "positive", "💩": "positive","😕": "nagative", "😱": "nagative", "😸": "positive",
        "😾": "nagative", "🚫": "nagative",  "🤬": "nagative","🧚": "positive", "🧡": "positive",'🐶':' positive ',
        '👎': ' nagative ', '😣': ' nagative ','✨': ' positive ', '❣': ' positive ','☀': ' positive ',
        '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
        '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' nagative ', '😢': ' nagative ',
        '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' nagative ', '😊': ' positive ',
        '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' nagative ', '😭': ' nagative ',
        '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
        '^^': ' positive ', '😨': ' nagative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
        '😖': ' nagative ', '😀': ' positive ', ':((': ' nagative ', '😡': ' nagative ', '😠': ' nagative ',
        '😒': ' nagative ', '🙂': ' positive ', '😏': ' nagative ', '😝': ' positive ', '😄': ' positive ',
        '😙': ' positive ', '😤': ' nagative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
        '✌': ' positive ', '💕': ' positive ', '😞': ' nagative ', '😓': ' nagative ', '️🆗️': ' positive ',
        '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
        '💓': ' positive ', '😐': ' nagative ', ':3': ' positive ', '😫': ' nagative ', '😥': ' nagative ',
        '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
        '😗': ' positive ', '🤔': ' nagative ', '😑': ' nagative ', '🔥': ' nagative ', '🙏': ' nagative ',
        '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
        '😚': ' positive ', '❌': ' nagative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
        '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
        '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
        '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
        '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
        '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ','☹': ' nagative ',  '💀': ' nagative ',
        '😔': ' nagative ', '😧': ' nagative ', '😩': ' nagative ', '😰': ' nagative ', '😳': ' nagative ',
        '😵': ' nagative ', '😶': ' nagative ', '🙁': ' nagative ',
        # Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ','hehe': ' positive ','hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' nagative ',' cc ': ' nagative ','cute': u' dễ thương ','huhu': ' nagative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
        # Dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',}

    for k, v in replace_list.items():
        text = text.replace(k, v)

    # Chuyển punctuation thành space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # Tokenize văn bản
    text = ViTokenizer.tokenize(text)
    texts = text.split()
    len_text = len(texts)

    # Remove nốt những ký tự thừa thãi
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻','')
    text = text.replace('_',' ')
    
    return text

# Hàm tiền xử lý dữ liệu đánh giá sản phẩm
def pre_process(comment):
    comment = ViTokenizer.tokenize(comment)
    tokens = remove_special_text(comment)
    tokens = tokens.lower()
    tokens = tokens.split()
    s = ""
    for label in tokens:
        if len(label) == 1:
            label = acronym_text_one_word(label)
        elif len(label) == 2:
            label = acronym_text_two_words(label)
        elif len(label) == 3 :
            label = acronym_text_three_words(label)
        elif len(label) == 4:
            label = acronym_text_four_words(label)
        else:
            label = acronym_text(label)
        s += label + " "
    tokens = s
    tokens = normalize_text(tokens)
    return " ".join(tokens.split())

# Đọc file CSV
input_file = 'comments_output.csv'  # Đường dẫn tới file CSV gốc
output_file = 'preprocessed_comments.csv'  # Đường dẫn tới file CSV mới

# Đọc dữ liệu từ file CSV
df = pd.read_csv(input_file)

# Xóa các dòng có giá trị NaN trong cột 'content'
df = df.dropna(subset=['content'])

# Xóa các dòng có giá trị trùng lặp trong cột 'content'
df = df.drop_duplicates(subset=['content'])

# Giả sử cột chứa các comment cần xử lý tên là 'comment'
# Thêm cột mới với kết quả xử lý
df['content'] = df['content'].apply(pre_process)

# Lưu kết quả vào file CSV mới
df.to_csv(output_file, index=False)
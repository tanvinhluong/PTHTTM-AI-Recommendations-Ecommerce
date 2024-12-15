from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql

# Sử dụng PyMySQL làm MySQLdb
pymysql.install_as_MySQLdb()

app = Flask(__name__)

# Load dữ liệu từ tệp CSV
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Cấu hình database
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost:3307/ecommerce_ptit"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your models class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your models class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

# gợi ý dựa trên nội dung của sản phẩm như tên và các tags
def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

# gợi ý dựa trên thói quen của user đó
def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # Create the user-item matrix
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)

    # Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = []

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item IDs of recommended items
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    # Get the details of recommended items
    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details.head(10)


# Hybrid Recommendations (Combine Content-Based and Collaborative Filtering)
def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    # Get content-based recommendations
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)

    # Get collaborative filtering recommendations
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)

    # Merge and deduplicate the recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()

    return hybrid_rec.head(10)
# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html')

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        # Kiểm tra thông tin đăng nhập của người dùng trong cơ sở dữ liệu
        user = Signin.query.filter_by(username=username).first()

        # Nếu không tìm thấy người dùng hoặc mật khẩu sai
        if user is None or user.password != password:
            return render_template('signin.html', error="Tên người dùng hoặc mật khẩu không đúng.")

        # Nếu đăng nhập thành công
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

        return render_template('index.html',
                               trending_products=trending_products.head(8),
                               truncate=truncate,
                               random_product_image_urls=random_product_image_urls,
                               random_price=random.choice(price),
                               signup_message='Đăng nhập thành công!')

    # Nếu là phương thức GET, trả về trang đăng nhập
    return render_template('index.html')
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        # Sở thích người dùng và các sản phẩm tương tự
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            print(content_based_rec)
            print(random_product_image_urls)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))
    return render_template('search.html')

@app.route("/collaborative", methods=['POST', 'GET'])
def collaborative_recommendations():
    if request.method == 'POST':
        user_id = int(request.form.get('user_id'))  # Nhận ID người dùng từ form
        nbr = int(request.form.get('nbr'))  # Nhận số lượng sản phẩm gợi ý từ form

        # Gọi hàm collaborative filtering để lấy các sản phẩm gợi ý
        collaborative_rec = collaborative_filtering_recommendations(train_data, user_id, top_n=nbr)

        if collaborative_rec.empty:
            message = "No recommendations available for this user."
            return render_template('main-collaborative.html', message=message)
        else:
            # Tạo danh sách hình ảnh ngẫu nhiên cho từng sản phẩm gợi ý
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(collaborative_rec))]
            print(collaborative_rec)
            print(random_product_image_urls)

            # Tạo giá ngẫu nhiên
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main-collaborative.html', collaborative_rec=collaborative_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))

    return render_template('search-user.html')


@app.route("/hybrid-recommendations", methods=['POST', 'GET'])
def hybrid_recommendations_route():
    if request.method == 'POST':
        user_id = int(request.form.get('user_id'))  # Nhận ID người dùng từ form
        item_name = request.form.get('item_name')  # Nhận tên sản phẩm từ form
        nbr = int(request.form.get('nbr'))  # Nhận số lượng sản phẩm gợi ý từ form

        # Gọi hàm hybrid recommendations để lấy các sản phẩm gợi ý
        hybrid_rec = hybrid_recommendations(train_data, user_id, item_name, top_n=nbr)

        if hybrid_rec.empty:
            message = "No recommendations available."
            return render_template('main-hydrid.html', message=message)
        else:
            # Tạo danh sách hình ảnh ngẫu nhiên cho từng sản phẩm gợi ý
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            print(hybrid_rec)
            print(random_product_image_urls)

            # Tạo giá ngẫu nhiên
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main-hydrid.html', hybrid_rec=hybrid_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))

    return render_template('search-hydrid.html')
if __name__=='__main__':
    app.run(debug=True)
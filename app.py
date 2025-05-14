import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib as joblib
# import requests
# from sklearn.metrics.pairwise import cosine_similarity

# set app config
st.set_page_config(page_title="Movies Recommendation", page_icon="🎬", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url(""); 
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# Load models and MovieDB
# df = pd.read_pickle("datasets/clean/streamlit_movies_df.pkl")  
# cosine_similarity_matrix = np.load('models/cosine_similarity_matrix.npy')
# tfidf_matrix = joblib.load('models/tfidf_matrix.joblib')
# tfidf_vector = joblib.load('models/tfidf_vectorizer.joblib')


# # Movie Title based Recommendation Function
# def get_recommendations(movie_title, matrix=cosine_similarity_matrix, n=5):

#     # get index from dataframe
#     index = df[df['title'] == movie_title].index[0]
    
#     # sort top n similar movies     
#     similar_movies = sorted(list(enumerate(matrix[index])), reverse=True, key=lambda x: x[1]) 
    
#     # extract names from dataframe and return movie names
#     recommendations = []
#     for i in similar_movies[1:n+1]:
#         recommendations.append(df.iloc[i[0]].title)
#     return recommendations
    
# # Keywords based Recommendation Function
# def get_recommendations_by_keyword(keywords, vector=tfidf_vector, matrix=tfidf_matrix, n=5):

#     keywords = keywords.split()
#     keywords = " ".join(keywords)
    
#     # transform the string to vector representation
#     key_matrix = vector.transform([keywords]) 
    
#     # compute cosine similarity    
#     result = cosine_similarity(key_matrix, matrix)

#     # sort top n similar movies   
#     similar_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1])
    
#     # extract names from dataframe and return movie names
#     recommendations = []
#     for i in similar_movies[1:n+1]:
#         recommendations.append(df.iloc[i[0]].title)
#     return recommendations

    
# # Fetching Poster
# def fetch_poster(movies):
#     ids = []
#     posters = []
#     for i in movies:
#         ids.append(df[df.title==i]['id'].values[0])
    
#     # print(movies[0])
#     # print(ids[0])

#     for i in ids:    
#         url = f"https://api.themoviedb.org/3/movie/{i}?api_key=5a2d8928f55e54df341caf236e1fc136"
#         data = requests.get(url)
#         data = data.json()
#         # print(data)
#         poster_path = data['poster_path']
#         # print(poster_path)
#         full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#         # print(full_path)
#         posters.append(full_path)
#     return posters


# posters = 0
# movies = 0

# Side Bar
# with st.sidebar:
#     st.header("Get Recommendations by 🍿")
#     search_type = st.radio("", ('Movie Title', 'Keywords'))

# # call functions based on selectbox
# if search_type == 'Movie Title': 
#     # print(search_type)
#     st.subheader("Select Movie 🌟")   
#     movie_name = st.selectbox('', df.title)
#     if st.button('Recommend 🚀'):
#         with st.spinner('Wait for it...'):
#             movies = get_recommendations(movie_name)
#             # print(movies[0])
#             posters = fetch_poster(movies)        
# else:
#     st.subheader('Enter Cast / Director / Producer / Genre  🌟')
#     keyword = st.text_input('', 'Christopher Nolan')
#     if st.button('Recommend 🚀'):
#         with st.spinner('Wait for it...'):
#             movies = get_recommendations_by_keyword(keyword)
#             # print(movies[0])
#             posters = fetch_poster(movies)


# display movies
# if movies:
#     num_movies = len(movies)
#     num_cols = min(5, num_movies)  # Limit columns to 5

#     cols = st.columns(num_cols, gap='medium')

#     max_poster_height = 200  # Set a maximum poster height

#     for i in range(num_movies):
#         with cols[i % num_cols]:  # Wrap around columns for more than 5 movies
#             with st.container():
#                 # st.text(movies[i])
#                 st.markdown(f"<h3 style='font-size: 16px; height: 50px;'>{movies[i]}</h3>", unsafe_allow_html=True) 
#                 if posters[i]:
#                     st.image(posters[i])
#                 else:
#                     st.empty() 

#     # Add CSS to set a fixed height for all containers
#     st.markdown(f"""
#     <style>
#     .st-container {{
#         height: {max_poster_height}px; 
#         display: flex;
#         flex-direction: column;
#         justify-content: center;
#         align-items: center;
#     }}
#     </style>
#     """, unsafe_allow_html=True)

#     st.markdown(f"""
#     <style>
#     body {{
#         background-color: #ffffff;
#     }}
#     </style>
#     """, unsafe_allow_html=True)

# else:
#     st.title('Hello Streamlit!')
#     st.write("This is a simple web app to display text.")

def main():
    st.title('Hello Streamlit!')
    st.write("This is a simple web app to display text.")

if __name__ == "__main__":
    main()

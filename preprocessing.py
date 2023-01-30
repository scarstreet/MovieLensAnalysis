import pandas as pd

# use this file for functions to read data from the ml- 100k file

'''
mainly, data to analyze from u.data, u.user, u.item
u.data  -   the ratings data. user rated item
            [userID|itemID|Rating|timestamp]
u.item  -   The information regarding the movies
            [movie id | movie title | release date | video release date | IMDb URL | ..genres 19 fields.. ]
u.user  -   Info about the users
            [user id | age | gender | occupation | zip code ]
'''

'''
u.occupation is to accompany u.user
u.genre is to accompany u.item
'''

# first, read the rating data
def readRatingData():
    rating_header = ["user_id", "item_id", "rating", "timestamp"]
    rating = pd.read_csv("ml-100k\\u.data", sep = '\t', header = None, names=rating_header)
    rating = rating.drop(['timestamp'],axis=1)
    return rating

def readItemData():
    movie_header = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL","unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime","Documentary", "Drama", "Fantasy", "Film-Noir","Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies = pd.read_csv("ml-100k\\u.item", sep = '|', header = None, encoding = 'latin1', names = movie_header)
    movies = movies.drop(columns=['video_release_date', "title", "release_date", "IMDb_URL"]) #the only columns that matter is just id and genres hahahaah
    return movies

def readUserData():
    user_header = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv("ml-100k\\u.user", sep = '|', header = None, names=user_header)

    occupation = pd.read_csv("u.occupation", header = None)
    occupation_list = occupation.values

    users["gender"].replace(['F', 'M'],[0, 1], inplace=True)
    users["occupation"].replace(occupation_list,list(range(0, len(occupation_list))), inplace=True)
    return users

def specifyByUserData(users,ratings,categ):
    # user based can be classified by "age", "gender", "occupation", "zip_code"
    # can specify what we wanna analyze from categ input
    # TODO - improve age by having an agespan
    user_header = ["user_id"].concat(categ)
    _user = users.loc[:,user_header]
    df = pd.merge(_user,ratings, on=['user_id'])
    return df




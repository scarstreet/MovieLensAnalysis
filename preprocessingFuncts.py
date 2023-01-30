import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import sklearn.metrics as metrics

# use this file for functions to read data from the ml- 100k file
# I just made this to separate data and throw away useless ones

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


def readRatingData(path="ml-100k\\u.data"):
    rating_header = ["user_id", "item_id", "rating", "timestamp"]
    rating = pd.read_csv(path, sep='\t',
                         header=None, names=rating_header)
    rating = rating.drop(['timestamp'], axis=1)
    return rating


def readItemData(path="ml-100k\\u.item"):
    movie_header = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies = pd.read_csv(path, sep='|',
                         header=None, encoding='latin1', names=movie_header)
    movies["release_date"] = movies["release_date"].map(
        lambda x: x[-4:] if type(x) == str else x)
    # the only columns that matter is just id and genres hahahaah
    movies = movies.drop(
        columns=['video_release_date', "IMDb_URL"])#changed release date to 'k, cause you're changing release date to year, and if it's dropped then we cannot change, hence program doesn't work
    movies = movies.rename(columns={"release_date": "year"})
    # temp = movies.copy()
    # temp['movies'] = pd.to_numeric(temp[""])
    movies["year"] = pd.to_numeric(movies["year"])
    movies["year_category"] = pd.cut(movies["year"], bins=[0, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000], labels=[0, 1, 2, 3, 4, 5, 6, 7])
    return movies
# readItemData()

def yearcateg():
     movies= readItemData()
     movies= movies[["item_id", "title","year"]]
    #  print(movies)
     count=pd.DataFrame()
     count["count"]=movies.groupby(["year"])["year"].count()
     count=count.reset_index()
     movies=pd.merge(movies,count)
     movies["year"]=pd.to_numeric(movies["year"])
     movies["year_category"]=pd.cut(movies["year"],bins=[0,1930,1940,1950,1960,1970,1980,1990,2000], labels=[1920,1930,1940,1950,1960,1970,1980,1990])
     return movies
# yearcateg()

def readUserData(path="ml-100k\\u.user"):
    user_header = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv(path, sep='|',
                        header=None, names=user_header)

    occupation = pd.read_csv("ml-100k\\u.occupation", header=None)
    occupation_list = occupation.values

    

    users["gender"].replace(['F', 'M'], [0, 1], inplace=True)
    users["occupation"].replace(occupation_list, list(
        range(0, len(occupation_list))), inplace=True)
    users["age_category"] = pd.cut(users["age"], bins = [0, 10, 20, 30, 40, 50, 60, 70, 80], labels=[0,1,2,3,4,5,6,7])
    #print(users["age_category"])
    return users


def getOccupationList(path="ml-100k\\u.occupation"):
    occupation_header = ["occupation"]
    rating = pd.read_csv(path,names=occupation_header)
    return rating

def getGenreList(pathitem="ml-100k\\u.item",pathgenre="ml-100k\\u.genre"):
    movie_header = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
         "unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime",
         "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
         "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies = pd.read_csv(pathitem, sep = '|', header = None, encoding = 'latin1', names = movie_header)
    movies = movies.drop(columns=['video_release_date'])
    genre = pd.read_csv(pathgenre, sep = '|', header = None)
    genre_list = genre[0].values
    movie_set_genre = movies[genre_list]
    genre_array = movie_set_genre.to_numpy()
    return genre_array

def categorySimilarity(occup,tick,string,shown,size=(20,20),threshold=30,setindex='item_id'):
    sim = occup.pivot_table(columns=string,index=setindex,values='rating').fillna(0) #Get the pivot table
    a = sim.corr(min_periods=threshold) #Get the correlation with threshold 30
    if(shown=='false'):
        return a
    plt.figure(figsize=size) #figure the size, default = 20,20, we can set it based on what we like
    plt.set_cmap('jet') #Set the color of the box
    plt.imshow(a) #Create the matrix table
    plt.colorbar() #Show the right side color
    plt.xticks(range(0,len(tick)),tick,rotation=-90) 
    plt.yticks(range(0,len(tick)),tick)
    if(string != 'year'):
        for i in a.columns:
            for j in a.columns:
                pass
                # plt.annotate(xy=(i,j),text=str(a[j][i].round(2)),va='center',ha='center') #setting the text in each matrix box
                plt.text(i,j,str(a[j][i].round(2)),va='center',ha='center') #setting the text in each matrix box
    plt.show() #show the plot
    # return sim

def getEssentials():
    users = readUserData()
    users = users.sort_values(by=['user_id','age']).reset_index().drop('index',axis=1)

    items = readItemData()
    items = items.sort_values(by=['year','item_id']).reset_index().drop('index',axis=1)
    items = items.dropna()

    ratings = readRatingData()
    ratings = ratings.sort_values(by=['user_id','item_id']).reset_index().drop('index',axis=1)
    return users,items,ratings

def getSimOccup(shown='false'):
    users,items,ratings = getEssentials()
    occup = specifyByUserData(users, ratings, ["occupation"])
    job = getOccupationList()
    occup = occup.drop('user_id',axis=1)
    occup = occup.groupby(by=['occupation','item_id']).mean()
    tick = job['occupation'].tolist()
    return categorySimilarity(occup,tick,'occupation',shown)

def getSimAgeCategory(shown='false'):
    users,items,ratings = getEssentials()
    occup = specifyByUserData(users, ratings, ["age_category"])
    occup = occup.drop('user_id',axis=1)
    occup = occup.groupby(by=['age_category','item_id']).mean()
    age = [f"{x*10+1} - {(x+1)*10}" for x in range(0,8)]
    return categorySimilarity(occup,age,'age_category',shown)

def getSimGender(shown='false'):
    users,items,ratings = getEssentials()
    occup = specifyByUserData(users, ratings, ["gender"])
    occup = occup.drop('user_id',axis=1)
    occup = occup.groupby(by=['gender','item_id']).mean()
    gender = ['female','male']
    return categorySimilarity(occup,gender,'gender',shown,size=(10,10))

def getSimItemYear(shown='false'):
    users,items,ratings = getEssentials()
    occup = saveyear = specifyByItemData(items, ratings, "year")
    occup = occup.drop('user_id',axis=1)
    occup = occup.groupby(by=['year','item_id']).mean()

    #saving only the year list
    saveyear = saveyear['year'].drop_duplicates().reset_index().drop('index',axis=1)
    saveyeartext = saveyear['year'].tolist()
    return categorySimilarity(occup,saveyeartext,'year',shown,size=(40,40),threshold=0)

def getSimUserYear(shown='false'):
    users,items,ratings = getEssentials()
    occup = saveyear = specifyByItemData(items, ratings, "year")
    occup = occup.drop('item_id',axis=1)
    occup = occup.groupby(by=['year','user_id']).mean()

    #saving only the year list
    saveyear = saveyear['year'].drop_duplicates().reset_index().drop('index',axis=1)
    saveyeartext = saveyear['year'].tolist()
    return categorySimilarity(occup,saveyeartext,'year',shown,size=(40,40),threshold=0,setindex='user_id')

def getSimUserYearCategory(shown='false'):
    users,items,ratings = getEssentials()
    occup = saveyear = specifyByItemData(items, ratings, "year_category")
    occup = occup.drop('item_id',axis=1)
    occup = occup.groupby(by=['year_category','user_id']).mean()

    #saving only the year list
    # saveyear = saveyear['year_category'].drop_duplicates().reset_index().drop('index',axis=1)
    saveyeartext = ["1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s"]
    return categorySimilarity(occup,saveyeartext,'year_category',shown,size=(40,40),threshold=0,setindex='user_id')

def getSimGenre(action="getSim"):
    genre_array = getGenreList()
    distance_matrix = metrics.pairwise_distances(genre_array,metric = 'jaccard') # ‘cosine’, ‘euclidean’, etc
    if(action=="getSim"):
        return distance_matrix
    plt.figure(figsize=(40,40))
    plt.imshow(distance_matrix)
    plt.set_cmap('jet') #Set the color of the box
    plt.colorbar() #Show the right side color
    plt.show()

# Best/worst ratings for user categs
def Unthresholduserdata(categ):
    rating=readRatingData()
    users=readUserData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    print(rating.sort_values("average_rating",ascending=False))
    rating=pd.merge(rating,users[["user_id","gender","occupation","age_category"]])
    rating=pd.merge(rating,movies[["item_id","title"]])
    rating=rating.drop(["user_id","rating"],axis=1)
    storedparameter=[]
    if categ=="gender":
    ######################Gender######################
        for a in range(2):
            parameterMax=rating[rating["gender"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["gender"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Gender {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Gender {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)

    ##################################################
    elif categ=="occupation":
    ######################occupation##################
        a=0
        for a in range(21):
            parameterMax=rating[rating["occupation"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["occupation"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Occupation {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Occupation {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)

    ##################################################
    elif categ=="age_group":
    ######################Age_group###################
        a=1
        for a in range(8):
            parameterMax=rating[rating["age_category"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["age_category"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Age Group {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Age Group {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)

    else:
         raise Exception(f"categ should be 'gender' or 'occupation' or 'age_group'; given {categ}")
    ##################################################
    return storedparameter

#Unweighteduserdata("age_group")

def Thresholduserdata(threshold, categ):
    rating=readRatingData()
    users=readUserData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    weight=pd.DataFrame()
    weight["count"]=rating.groupby(["item_id"])["item_id"].count()
    weight=weight.reset_index()
    
    filter=(weight["count"]>=threshold)
    weight=weight[filter]
    rating=pd.merge(rating,weight).sort_values(by=["count"],ascending=True)
    print(rating.sort_values("average_rating",ascending=False))
    rating=pd.merge(rating,users[["user_id","gender","occupation","age_category"]])
    rating=pd.merge(rating,movies[["item_id","title"]])
    rating=rating.drop(["user_id","rating","count"],axis=1)
    #print(rating)
    parameterMax=0
    parameterMin=0
    storedparameter=[]
    if categ=="gender":
    ######################Gender######################
        for a in range(2):
            parameterMax=rating[rating["gender"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["gender"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Gender {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Gender {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)

    ##################################################
    elif categ=="occupation":
    ######################occupation##################
        a=0
        for a in range(21):
            parameterMax=rating[rating["occupation"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["occupation"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Occupation {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Occupation {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
    ##################################################
    elif categ=="age_group":
    ######################Age_group###################
        for a in range(8):
            parameterMax=rating[rating["age_category"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["age_category"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            parameterMax=parameterMax.style.set_caption(f"Age group {a} Max:")
            parameterMin=parameterMin.style.set_caption(f"Age group {a} Min:")
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
    else:
         raise Exception(f"categ should be 'gender' or 'occupation' or 'age_group'; given {categ}")
    ##################################################
    #print(storedparameter)

    return storedparameter
#Weighteduserdata(30,"gender")
# Unweighteduserdata("age_group")

def Thresholditemdata(threshold,moviegenre):
    rating=readRatingData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    weight=pd.DataFrame()
    weight["count"]=rating.groupby(["item_id"])["item_id"].count()
    weight=weight.reset_index()
    filter=(weight["count"]>=threshold)
    weight=weight[filter]
    rating=pd.merge(rating,weight).sort_values(by=["count"],ascending=True)
    rating=pd.merge(rating,movies)
    rating=rating.drop(["user_id","rating","count"],axis=1)
    moviedict=["unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    final=[]
    if moviegenre in moviedict:
        if moviegenre == "unknown":
            final=1
            #return print("Weighted data for Unknown genre does not exist")
            
        else:
            testmax=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            testmax=testmax[["item_id","title"]]
            
            testmin=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=True).head()
            testmin=testmin[["item_id","title"]]
            testmax=testmax.style.set_caption(f"{moviegenre} Max:")
            testmin=testmin.style.set_caption(f"{moviegenre} Min:")
            final.append(testmax)
            final.append(testmin)
    else:
        raise Exception("rewrite the genre")
    if moviegenre== "unknown":
        return print("Weighted data for Unknown genre does not exist")
    else:
        return final

def Unthresholditemdata(moviegenre):
    rating=readRatingData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    rating=pd.merge(rating,movies)
    rating=rating.drop(["user_id","rating"],axis=1)
    moviedict=["unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    final=[]
    if moviegenre in moviedict:
            testmax=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            testmax=testmax[["item_id","title"]]
            
            testmin=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=True).head()
            testmin=testmin[["item_id","title"]]
            testmax=testmax.style.set_caption(f"{moviegenre} Max:")
            testmin=testmin.style.set_caption(f"{moviegenre} Min:")
            final.append(testmax)
            final.append(testmin)
    else:
        raise Exception("rewrite the genre")
    return final

def specifyByUserData(users, ratings, categ):
    # user based can be classified by "age", "gender", "occupation", "zip_code"
    # can specify what we wanna analyze from categ input
    user_header = ["user_id"]
    user_header.extend(categ)
    _user = users.loc[:, user_header]
    df = pd.merge(_user, ratings, on=['user_id'])
    return df

def specifyByItemData(items, ratings, categ):
    # categ only 2, genres or year
    item_header = ["item_id"]
    if categ == "year":
        item_header.append("year")
    
    elif categ=="year_category":
        item_header.append("year_category")

    elif categ == "genres":
        item_header.extend(["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    elif categ == "all":
        item_header.append("year")
        item_header.append("year_category")
        item_header.extend(["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    else:
        raise Exception(
            "category can only be strings \"year\", \"genres\" or \"all\"")
    # print(item_header)
    # display(items)
    _item = items.loc[:, item_header]
    df = pd.merge(_item, ratings, on=['item_id'])
    return df

# TODO - group zipcodes by this lib from https://www.zipcode.com.ng/2022/06/list-of-5-digit-zip-codes-united-states.html - steven
# REMEMBER GUYS, read table from html in pandas exist. no need for awesome webcrawling acrobatics


''' similarities '''
# TODO - connect the ratings ID to item
# TODO - compare user info with genres

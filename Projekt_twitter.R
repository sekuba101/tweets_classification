library(twitteR)
library(ROAuth)
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(purrrlyr)
library(base64enc)
library(ggplot2)
#function for converting symbols
conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")


twitterTexts <- read_csv("/home/sekuba101/rExercises/Machine Learning RRRR/data_set/tweetsData.csv",
                         col_names = c('sentiment', 'id', 'date', 'query', 'user', 'text'))
#
twitterTexts<-dmap_at(twitterTexts,'text', conv_fun)

#zmiana wartości 4 na 1 w rubryce 'sentiment'
twitterTexts<-mutate(twitterTexts, sentiment = ifelse(sentiment == 0, 0, 1))

#przygotowanie danych - podział na zbiory uczący i testowy. Moduł createDataPartition wykorzystuje crossvalidation, dzieki temu mozemy
#poddzielic dane na k grup. W obecnej chwili wystarczy nam jeden podział(times=1). 

#generujemy losową liczbę - dzięki temu uzyskamy za każdym razem inny podział danych treningowych
set.seed(5144)  
#generujemy losowo indeksy ze zioru twitterTexts które użyjemy w klasyfikacji jako zbiór treningowy - jako tabele(list = FALSE)
trainIndex <- createDataPartition(twitterTexts$sentiment, p = 0.7, 
                                  list = FALSE, 
                                  times = 1,  #ilość podziałow
                                  )

tweets_train <- twitterTexts[trainIndex, ]
tweets_test <- twitterTexts[-trainIndex, ]

#funkcja przygotowujaca zmieniajaca wszystkie duze litery w małe
prep_fun <- tolower
#tokenizacja sprawia, że tekst dzielony jest na słowa, czyli z kilkudziesięciu znaków jesteśmy w stanie stworzyć słowa na podstawie znajdujacych sie miedzy nimi spacji
tok_fun <- word_tokenizer 

it_train <- itoken(tweets_train$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   ids = tweets_train$id,
                   progressbar = TRUE)
it_test <- itoken(tweets_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = tweets_test$id,
                  progressbar = TRUE)
# creating vocabulary and document-term matrix
#tworzymy słownik 
#słownik określa funkcją ilości wystąpień słów w tekście
vocab <- create_vocabulary(it_train) #this function collects unique terms and corresponding statistics.
print('ilość różnych słów w zbioerze treningowym to:')
print(length(vocab$term))
print(vocab) #doc count - różne sekwencje w których wystepuje słowo term_count- całkowita ilość wystąpień słowa
vectorizer <- vocab_vectorizer(vocab) #This function creates an object (closure) which defines on how to transform list of tokens into vector space - i.e. how to map words to indices. It supposed to be used only as argument to create_dtm, create_tcm, create_vocabulary.
#This is a high-level function for creating a document-term matrix
# tworzymy macierz występowania słowa w każdym zdaniu patrz general concept https://en.wikipedia.org/wiki/Document-term_matrix
dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)

# define  najpierw twor
tfidf <- TfIdf$new()
# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- fit_transform(dtm_test, tfidf)
# train the model
t1 <- Sys.time()
#Does k-fold cross-validation for glmnet, produces a plot, and returns a value for lambda (and gamma if relax=TRUE)
#https://cran.r-project.org/web/packages/glmnet/index.html
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = tweets_train[['sentiment']], 
                               family = 'binomial', 
                               # L1 penalty
                               alpha = 1,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 5-fold cross-validation
                               nfolds = 5,
                               # high value is less accurate, but has faster training
                               thresh = 1e-2,
                               # again lower number of iterations for faster training
                               maxit = 1e2)
print(difftime(Sys.time(), t1, units = 'mins'))
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
glmnet:::auc(as.numeric(tweets_test$sentiment), preds)

# save the model for future usage
saveRDS(glmnet_classifier, 'glmnet_classifier.RDS')



### pobieranie i analiza tweetów ###
download.file(url = "http://curl.haxx.se/ca/cacert.pem",
              destfile = "cacert.pem")
consumer_key <- 'pz3AeqUFxLJUAu291gsJ6bwjW'
consumer_secret <- '6ztcmOU8KFHbQTPvbbXt6cC2BySwch9mTxYCRUNwmJY9mRdUog'
access_token <- '1357062011033681935-Io7jRytMN0SJO0vnpzpjMG62wCCgBH'
access_secret <- 'CYarAan8dkLf6EbeVCNCkW4e4EidWgcIjyoesjNs6lGMR'

twitteR:::setup_twitter_oauth(consumer_key, # api key
                    consumer_secret, # api secret
                    access_token, # access token
                    access_secret # access token secret
)
#This function will take a list of objects from a single twitteR class and return a data.frame version of the members
df_tweets <- twitteR:::twListToDF(twitteR:::searchTwitter('setapp OR #setapp', n = 1000, lang = 'en'))
  # converting some symbols
  dmap_at('text', conv_fun)
# preprocessing and tokenization
it_tweets <- text2vec:::itoken(df_tweets$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = df_tweets$id,
                    progressbar = TRUE)
# creating vocabulary and document-term matrix
dtm_tweets <- text2vec:::create_dtm(it_tweets, vectorizer)
# transforming data with tf-idf
dtm_tweets_tfidf <- mlapi:::fit_transform(dtm_tweets, tfidf)
# loading classification model
glmnet_classifier <- readRDS('glmnet_classifier.RDS')
# predict probabilities of positiveness
preds_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = 'response')[ ,1]
# adding rates to initial dataset
df_tweets$sentiment <- preds_tweets





# color palette
cols <- c("#ce472e", "#f05336", "#ffd73e", "#eec73a", "#4ab04a")
set.seed(932)
samp_ind <- sample(c(1:nrow(df_tweets)), nrow(df_tweets) * 0.1) # 10% for labeling
# plotting
ggplot2:::ggplot(df_tweets, aes(x = created, y = sentiment, color = sentiment)) +
  theme_minimal() +
  scale_color_gradientn(colors = cols, limits = c(0, 1),
                        breaks = seq(0, 1, by = 1/4),
                        labels = c("0", round(1/4*1, 1), round(1/4*2, 1), round(1/4*3, 1), round(1/4*4, 1)),
                        guide = guide_colourbar(ticks = T, nbin = 50, barheight = .5, label = T, barwidth = 10)) +
  geom_point(aes(color = sentiment), alpha = 0.8) +
  geom_hline(yintercept = 0.65, color = "#4ab04a", size = 1.5, alpha = 0.6, linetype = "longdash") +
  geom_hline(yintercept = 0.35, color = "#f05336", size = 1.5, alpha = 0.6, linetype = "longdash") +
  geom_smooth(size = 1.2, alpha = 0.2) +
  ggrepel:::geom_label_repel(data = df_tweets[samp_ind, ],
                   aes(label = round(sentiment, 2)),
                   fontface = 'bold',
                   size = 3,
                   max.iter = 100) +
  theme(legend.position = 'bottom',
        legend.direction = "horizontal",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 20, face = "bold", vjust = 2, color = 'black', lineheight = 0.8),
        axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.y = element_text(size = 8, face = "bold", color = 'black'),
        axis.text.x = element_text(size = 8, face = "bold", color = 'black')) +
  ggtitle("Analiza tweetów (prawdopodobieństwo bycia pozytywnym)")

#https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization
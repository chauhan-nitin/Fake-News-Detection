############################################################################
#------------------------- LOADING THE DATASET ----------------------------#
############################################################################
rm(list=ls(all=TRUE))
setwd("C:/Users/obc1/Desktop/Thesis (Readme)/Fake News Thesis")
#setwd("C:/Users/nc15/Desktop/Fake News")

require(data.table)

#Loading the train data using method fread (Large size or tsv format)
trainnews<-fread("train.tsv", header = "auto",na.strings = "")
testnews<-fread("test.tsv", header = "auto",na.strings = "")
validnews<-fread("valid.tsv", header = "auto",na.strings = "")

merge_news <- rbind(trainnews,testnews,validnews)

#Assigning colnames to the attributes
names(merge_news) <- c("ID","Label","Statement","Subject","Speaker","Job-Title","State","Party","Barely-True","False-Counts", "Half-True","Mostly-True","Pants-Fire","Context")

#Cleaning ID attribute
merge_news$ID <- gsub(pattern = "\\.json$", "", merge_news$ID)

#Checking the structure and understanding the variables
str(merge_news)
summary(merge_news)
sapply(merge_news, function(x) sum(is.na(x)))
merge_news$ID <- as.integer(merge_news$ID)
merge_news$Label <- as.factor(merge_news$Label)
merge_news$Subject_list <- as.list(strsplit(merge_news$Subject, ","))
merge_news$Speaker <- as.factor(merge_news$Speaker)
merge_news$`Job-Title` <- as.factor(merge_news$`Job-Title`)
merge_news$State <- as.factor(merge_news$State)
merge_news$Party <- as.factor(merge_news$Party)

attach(merge_news)

#################################################################
#--------- Data Preprocessing and Feature Engineering ----------#
#################################################################


################################
#----------- LABEL ------------#
################################

#Option 1: Multinomial Classification
summary(merge_news$Label)
library(ggplot2)
levels(merge_news$Label)
qplot(merge_news$Label, fill=Label)
merge_news$Label = factor(merge_news$Label,levels(merge_news$Label)[c(5,2,1,3,4,6)])
levels(merge_news$Label)
prop.table(table(merge_news$Label))
qplot(merge_news$Label, fill=Label)
str(merge_news$Label)

#Option 2: Three-class Classification (After Option 1)
#Reducing and Transforming into 3 class problem
levels(merge_news$Label) <- c("false", "false", "half-true", "half-true", "true","true")
levels(merge_news$Label)
qplot(merge_news$Label, fill=Label)
summary(merge_news$Label)
prop.table(table(merge_news$Label))
str(merge_news$Label)

#Option 3: Binary Classification
levels(merge_news$Label) <- c("false", "false", "false", "true", "true","true")
levels(merge_news$Label)
summary(merge_news$Label)
merge_news$Label <- factor(merge_news$Label, levels=c("false","true"), labels=c(0,1))
summary(merge_news$Label)
prop.table(table(merge_news$Label))
qplot(merge_news$Label, fill=Label)
str(merge_news$Label)

#Option 4: Ordinal Classification (Package Ordinal)
str(merge_news$Label)
merge_news$Label <- factor(Label, levels=c("pants-fire", "false", "barely-true","half-true","mostly-true","true"), ordered = TRUE)
merge_news$Label
levels(merge_news$Label)
summary(merge_news$Label)


merge_news$TextLength <- nchar(merge_news$Statement)
summary(merge_news$TextLength)
TextLength <- merge_news$TextLength
TextLength[1:50]

ggplot(merge_news, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")



################################
#---------- SUBJECT -----------#
################################

str(merge_news$Subject)
#merge_news$Subject_list <- as.list(strsplit(merge_news$Subject, ","))
str(merge_news$Subject_list)
library(tm)
myVCorpus <- VCorpus(VectorSource(merge_news$Subject_list))
myTdm <- DocumentTermMatrix(myVCorpus)
#Grid Search for Hyperparameter Optimization
?removeSparseTerms
#We're referring to most common top 10 subjects discussed
sparse = removeSparseTerms(myTdm,0.95) #Number of terms retained 10
rm(myVCorpus)
as.matrix(sparse)
v <- as.matrix(sparse)
dim(v)
class(v)
v1 <- colSums(v)
length(v1)
sort(v1, decreasing = TRUE)[1:10]

myCorpus <- Corpus(VectorSource(merge_news$Subject_list))
myTDM <- TermDocumentMatrix(myCorpus, control = list(minWordLength = 645))
tdm <- as.matrix(myTDM)
dim(tdm); tdm <- t(tdm)
View(tdm[1:10,1:181])
tdm <- as.data.frame(tdm)
names(tdm)
#n-1 dummy variables. 10 dummy variables. If all are zero values then subject is none
tdm <- subset(tdm, select=c(economy,health,taxes,federal,education,jobs,state,candidates,elections,immigration))
head(tdm,5)


################################
#---------- SPEAKER -----------#
################################
str(merge_news$Speaker)
sort(summary(merge_news$Speaker)[1:10],decreasing = T)
prop.table(table((merge_news$Speaker)[1:10]))
var <- c("barack-obama","donald-trump","hillary-clinton","mitt-romney","john-mccain","scott-walker","chain-email","rick-perry","marco-rubio","rick-scott")
str(merge_news$Speaker)
length(var);rev(var)

for(i in rev(var)){
  merge_news$Speaker <- relevel(merge_news$Speaker, i)
}
#Grouping the remaining speakers as 'others'
summary(levels(merge_news$Speaker))
str(merge_news$Speaker)
levels(merge_news$Speaker) <- c("barack-obama","donald-trump","hillary-clinton","mitt-romney","john-mccain","scott-walker","chain-email","rick-perry","marco-rubio","rick-scott", rep("others",3308))
qplot(merge_news$Speaker)
#One-hot Encode the Variable
# Make democrat first
merge_news$Speaker <- relevel(merge_news$Speaker, "others")
merge_news$Speaker
qplot(merge_news$Speaker)
library(mlr)
?createDummyFeatures
x <- createDummyFeatures(merge_news$Speaker, method = "reference")
#n-1 dummy variables. Taking 'others' as reference. 10 dummy variables for 11 levels
dim(x)


################################
#------------ PARTY -----------#
################################

str(merge_news$Party)
sort(summary(merge_news$Party)[1:24], decreasing = T)
prop.table(table(merge_news$Party))
levels(merge_news$Party)
require(ggplot2)
qplot(merge_news$Party)
# Make democrat first
merge_news$Party <- relevel(merge_news$Party, "democrat")
merge_news$Party
# 24 Levels: democrat activist business-leader columnist ... tea-party-member
# Make republican first
merge_news$Party <- relevel(merge_news$Party, "republican")
merge_news$Party
qplot(merge_news$Party)
levels(merge_news$Party) <- c("republican", "democrat", rep("none",22))
str(merge_news$Party)
qplot(merge_news$Party, fill=Party)
prop.table(table(merge_news$Party))
party <- createDummyFeatures(merge_news$Party)#One-hot encoding
str(party)

################################
#----------- STATE ------------#
################################

str(merge_news$State)
sort(summary(merge_news$State), decreasing = T)
#Replace the missing values in State with Mode(most frequent occuring) since it's categorical
which(is.na(merge_news$State))
merge_news$State[is.na(merge_news$State)] <- "Texas"
sort(prop.table(table(merge_news$State)), decreasing = T)[1:5]
#Taking upto 5% of levels proportion in the total data
names(sort(prop.table(table(merge_news$State)), decreasing = T)[1:5])
var <- c("Texas","Florida","Wisconsin","New York","Illinois")
str(merge_news$State)
length(var);rev(var)
for(i in rev(var)){
  merge_news$State <- relevel(merge_news$State, i)
}
#Grouping the remaining speakers as 'others'
summary(levels(merge_news$State))
str(merge_news$State)
levels(merge_news$State) <- c("Texas","Florida","Wisconsin","New York","Illinois",rep("Rest",67))
qplot(merge_news$State)
#One-hot Encode the Variable
# Make democrat first
merge_news$State <- relevel(merge_news$State, "Rest")
merge_news$State
qplot(merge_news$State)
library(mlr)
?createDummyFeatures
y <- createDummyFeatures(merge_news$State, method = "reference")
#n-1 dummy variables. Taking 'others' as reference. 10 dummy variables for 11 levels
dim(y)



################################
#--------- STATEMENT ----------#
################################
str(merge_news$Statement)
merge_news$Statement[1]
library(quanteda)
help(package = "quanteda")
# Tokenize Statement.
train.tokens <- tokens(merge_news$Statement, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE,
                       remove_separators = TRUE)
# Take a look at a specific SMS message and see how it transforms.
train.tokens[[1]]
# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[1]]
#Remove Built in Stopwords in English
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[1]]
# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[1]]
# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE, verbose = FALSE)
dim(train.tokens.dfm)
docfreq(train.tokens.dfm[1:10,1:50])
#Option 1: Trim the dataset based upon sparsity removal (hyperparamet)
#new_dfm <- dfm_trim(train.tokens.dfm, sparsity = 0.975)
#dim(new_dfm)
#train.tokens.matrix <- as.matrix(new_dfm)
#train.tokens.df <- cbind(Label = merge_news$Label, data.frame(new_dfm))

# Transform to a matrix and inspect.
#train.tokens.matrix <- as.matrix(new_dfm)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:53])
dim(train.tokens.matrix)
# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]
#train.tokens.df <- cbind(Label = merge_news$Label, data.frame(new_dfm))
train.tokens.df <- cbind(Label = merge_news$Label, data.frame(train.tokens.dfm))
# Often, tokenization requires some additional pre-processing
names(train.tokens.df)[c(1,14,26,45)]
# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))
dim(train.tokens.df)
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}
# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}
# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}
# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)
# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

gc()
# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
length(incomplete.cases) #number of rows
# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf) #12836x8528
sum(which(!complete.cases(train.tokens.tfidf)))

# Make a clean data frame using the same process as before.
statement <- cbind(Label = merge_news$Label, TextLength, data.frame(train.tokens.tfidf))
View(train.tokens.tfidf[1:25, 1:25])
gc()

# We'll leverage the irlba package for our singular value 
# decomposition (SVD). The irlba package allows us to specify
# the number of the most important singular vectors we wish to
# calculate and retain for features.
library(irlba)
# Time the code execution
start.time <- Sys.time()
# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA). #300 hyperparameter
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)
# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time #12mins
# Take a look at the new feature data up close.
View(train.irlba$v)
# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document
# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]

# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
statement.svd <- data.frame(Label = merge_news$Label, TextLength, train.irlba$v)
rm(train.svd)
View(statement.svd[1:10,1:20])



################################
#---------- CONTEXT -----------#
################################

merge_news$Context[1]
# Tokenize Statement.
train.tokens <- tokens(merge_news$Context, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE,
                       remove_separators = TRUE)

# Take a look at a specific SMS message and see how it transforms.
train.tokens[[1]]
# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[1]]
#Remove Built in Stopwords in English
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[1]]
# Bi-gram
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)
train.tokens[[1]]

# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE, verbose = FALSE)
dim(train.tokens.dfm)
docfreq(train.tokens.dfm[1:10,1:50])
#Option 1: Trim the dataset based upon sparsity removal (hyperparamet)
#new_dfm <- dfm_trim(train.tokens.dfm, sparsity = 0.975)
#dim(new_dfm)
#train.tokens.matrix <- as.matrix(new_dfm)
#train.tokens.df <- cbind(Label = merge_news$Label, data.frame(new_dfm))

# Transform to a matrix and inspect.
#train.tokens.matrix <- as.matrix(new_dfm)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:53])
dim(train.tokens.matrix)
# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]
#train.tokens.df <- cbind(Label = merge_news$Label, data.frame(new_dfm))
train.tokens.df <- cbind(Label = merge_news$Label, data.frame(train.tokens.dfm))
# Often, tokenization requires some additional pre-processing
names(train.tokens.df)[c(1,14,26,45)]
# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))
dim(train.tokens.df)
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}
# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}
# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}
# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)
# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

gc()
# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
length(incomplete.cases) #number of rows
# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

# We'll leverage the irlba package for our singular value 
# decomposition (SVD). The irlba package allows us to specify
# the number of the most important singular vectors we wish to
# calculate and retain for features.
library(irlba)
# Time the code execution
start.time <- Sys.time()
# Perform SVD. Specifically, reduce dimensionality down to 300 columns
# for our latent semantic analysis (LSA). #300 hyperparameter
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)
# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time #5.8mins
# Take a look at the new feature data up close.
View(train.irlba$v)
# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
#
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document
# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]

# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).
context.svd <- data.frame(Label = merge_news$Label, train.irlba$v)
((context.svd[1:10,1:5]))


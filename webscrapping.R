library(rvest)
library(curl)
#Creating a new data file containing only statements for web scraping
#c <- merge_news$Statement
#str(c)
#c[1]
#write.csv(c,file="Data.csv",row.names = FALSE)

c <- read.csv(file = "Data.csv", header = TRUE)
c$x <- as.character(c$x)
str(c$x)

# Learning how Web Scraping works in R using rvest package
#Generating the URL for First Statement
url <- URLencode(paste0("https://www.google.com/search?q=",c$x[1]))
#Reading the webpage
webpage <- read_html(url)
#Using CSS selectors to scrap the rankings section
rank_data_html <- html_nodes(webpage,'#resultStats')
#Converting the ranking data to text
rank_data <- html_text(rank_data_html)


#-------------- SEARCH RESULTS ----------------#
#To iterate it for all the Statements in the dataset, we make rank_data a vector
rank_data <- c()
for(i in 11868:12836){
  #Generating the URL for each Statement
  url <- URLencode(paste0("https://www.google.com/search?q=",c$x[i]))
  #Reading the contents of HTML page
  webpage <- read_html(url)
  #Using CSS selectors to scrap the rankings section
  rank_data_html <- html_nodes(webpage,'#resultStats')
  #Converting the ranking data to text
  rank_data[i] <- html_text(rank_data_html)
  #Creating a delay system while scraping each Statement to avoid connection error
  Sys.sleep(sample(20, 1) * 0.2)
}

#Let's have a look at the rankings
head(rank_data)
#Let's have a look at the number of rankings element
length(rank_data)

resultdata <- rank_data[1:12836]
resultdata
head(resultdata)
newfile <- gsub(pattern = "^About ", "", resultdata)
newfile <- gsub(pattern = " results$", "", newfile)
newfile <- as.numeric(gsub(",","",newfile))
newfile
sum(is.na(newfile))
newfile[is.na(newfile)] <- 0
options(scipen = 999)
searchresults <- data.frame(Label = merge_news$Label[1:12836],Result =newfile)
plot(searchresults$Label,searchresults$Result)

library(vegan)
boxplot(newfile) # Check any outlier and handle them
max(newfile)
class(newfile)
head(newfile)
newfile[1204] <- 0
dataStd. <- decostand(newfile,"range") #Range method = (M-Mmin)/(Mmax-Mmin)
merge_news$Search <- dataStd.

#-------------- SOURCE OF NEWS ----------------#
#To read the sources of news
page_data <- c()
for(i in 1:2){
  #Generating the URL for each Statement
  url <- URLencode(paste0("https://www.google.com/search?q=",c$x[i]))
  #Reading the contents of HTML page
  webpage <- read_html(url)
  #Using CSS selectors to scrap the rankings section
  rank_data_html <- html_nodes(webpage,'.iUh30')
  #Converting the ranking data to text
  page_data[i] <- html_text(rank_data_html)
  #Creating a delay system while scraping each Statement to avoid connection error
  Sys.sleep(sample(20, 1) * 0.2)
}


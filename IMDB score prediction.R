library(VIM)
library(data.table)
library(formattable)
library(GGally)
library(carData)
library(car)
library(dplyr)


IMDB <- read.csv("Desktop/movie_metadata.csv")

colSums(sapply(IMDB, is.na))

IMDB <- IMDB[!is.na(IMDB$gross), ]
IMDB <- IMDB[!is.na(IMDB$budget), ]

IMDB$facenumber_in_poster[is.na(IMDB$facenumber_in_poster)] <- round(mean(IMDB$facenumber_in_poster, na.rm = TRUE))
IMDB[,c(5,6,8,13,24,26)][IMDB[,c(5,6,8,13,24,26)] == 0] <- NA
IMDB$num_critic_for_reviews[is.na(IMDB$num_critic_for_reviews)] <- round(mean(IMDB$num_critic_for_reviews, na.rm = TRUE))
IMDB$duration[is.na(IMDB$duration)] <- round(mean(IMDB$duration, na.rm = TRUE))
IMDB$director_facebook_likes[is.na(IMDB$director_facebook_likes)] <- round(mean(IMDB$director_facebook_likes, na.rm = TRUE))
IMDB$actor_3_facebook_likes[is.na(IMDB$actor_3_facebook_likes)] <- round(mean(IMDB$actor_3_facebook_likes, na.rm = TRUE))
IMDB$actor_1_facebook_likes[is.na(IMDB$actor_1_facebook_likes)] <- round(mean(IMDB$actor_1_facebook_likes, na.rm = TRUE))
IMDB$cast_total_facebook_likes[is.na(IMDB$cast_total_facebook_likes)] <- round(mean(IMDB$cast_total_facebook_likes, na.rm = TRUE))
IMDB$actor_2_facebook_likes[is.na(IMDB$actor_2_facebook_likes)] <- round(mean(IMDB$actor_2_facebook_likes, na.rm = TRUE))
IMDB$movie_facebook_likes[is.na(IMDB$movie_facebook_likes)] <- round(mean(IMDB$movie_facebook_likes, na.rm = TRUE))

IMDB <- IMDB %>% 
  mutate(profit = gross - budget,
         return_on_investment_perc = (profit/budget)*100)

genres.df <- as.data.frame(IMDB[,c("genres", "imdb_score")])
genres.df$Action <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Action") 1 else 0)
genres.df$Adventure <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Adventure") 1 else 0)
genres.df$Animation <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Animation") 1 else 0)
genres.df$Biography <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Biography") 1 else 0)
genres.df$Comedy <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Comedy") 1 else 0)
genres.df$Crime <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Crime") 1 else 0)
genres.df$Documentary <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Documentary") 1 else 0)
genres.df$Drama <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Drama") 1 else 0)
genres.df$Family <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Family") 1 else 0)
genres.df$Fantasy <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Fantasy") 1 else 0)
genres.df$`Film-Noir` <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Film-Noir") 1 else 0)
genres.df$History <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "History") 1 else 0)
genres.df$Horror <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Horror") 1 else 0)
genres.df$Musical <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Musical") 1 else 0)
genres.df$Mystery <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Mystery") 1 else 0)
genres.df$News <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "News") 1 else 0)
genres.df$Romance <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Romance") 1 else 0)
genres.df$`Sci-Fi` <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Sci-Fi") 1 else 0)
genres.df$Short <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Short") 1 else 0)
genres.df$Sport <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Sport") 1 else 0)
genres.df$Thriller <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Thriller") 1 else 0)
genres.df$War <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "War") 1 else 0)
genres.df$Western <- sapply(1:length(genres.df$genres), function(x) if (genres.df[x,1] %like% "Western") 1 else 0)
means <- rep(0,23)
for (i in 1:23) {
  means[i] <- mean(genres.df$imdb_score[genres.df[i+2]==1])
}
barplot(means, main = "IMDB scores for different genres", horiz=TRUE, xlab = "score", ylab = "genres")


ggplot(aes(x = title_year, y = imdb_score), data = IMDB) +
  geom_jitter(alpha = 1/4) + 
  geom_smooth(method = "auto") +
  ggtitle("IMDB scores for title_year")

ggplot(aes(x=budget, y = imdb_score), data = subset(IMDB, budget < 75000000),!is.na(budget)) +
  geom_jitter(alpha = 1/4) +
  geom_smooth(method = "auto")+
  ggtitle("IMDB scores for budget")+
  labs(x = "Budget", y = "imdb_score")
ggplot(aes(x=gross, y = imdb_score), data = subset(IMDB, gross < 500000000),!is.na(gross)) +
  geom_jitter(alpha = 1/4) +
  geom_smooth(method = "auto")+
  ggtitle("IMDB scores for gross")+
  labs(x = "Gross", y = "imdb_score")

ggplot(aes(x = duration, y = imdb_score), data = subset(IMDB, duration < 200)) +
  geom_jitter(alpha = 1/4) + 
  geom_smooth(method = "auto")+
  ggtitle("IMDB scores for duration")


train.index <- sample(row.names(IMDB), dim(IMDB)[1]*0.7)
train <- IMDB[train.index, ]
test <- IMDB[-train, ]


set.seed(2)
lmfit = lm(imdb_score~.,data=train)
pred <- predict(lmfit,test)
mean((test$imdb_score-pred)^2)


library(rpart)
set.seed(3)
m.rpart <- rpart(imdb_score~.,data=train)
p.rpart <- predict(m.rpart,test)
mean((test$imdb_score-p.rpart)^2)

              
              
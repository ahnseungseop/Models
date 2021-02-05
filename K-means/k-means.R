# load dataset
kmeans_iris <-iris
head(kmeans_iris)

# delete target(species) column
kmeans_iris$Species <- NULL
head(kmeans_iris)

# Make K-means cluster using K=3
clusters<-kmeans(kmeans_iris,3)
clusters

## 1st cluster has 50 datas 2nd is 38, 3rd is 62

# compare cluster label to species column
table(iris$Species, clusters$cluster)

## rows mean real label, columns mean label made by k-means clustering

# Visualize
plot(kmeans_iris[c("Sepal.Length","Sepal.Width")],
     
     col = clusters$cluster,pch = c(15,16,17)[as.numeric(clusters$cluster)])

points(clusters$centers[,c("Sepal.Length","Sepal.Width")], col = 1:3, pch=8, cex=5)

## star means average point in each cluster
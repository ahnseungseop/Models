# load packages

library(dplyr)
library(gridExtra)
getwd()
setwd("C:/Users/inolab/Desktop/개인연구/R/머신러닝")

########### utility function : vector change to list

list.append <- function(mylist, ...){
  mylist<-c(mylist, list(...))
  return(mylist)
}

###################### 1단계 : read data

get_transaction_dataset <- function(filename){
  df <- read.csv(filename, header = FALSE)
  dataset <- list()
  for (index in seq(nrow(df))){
    transaction.set <- as.vector(unlist(df[index,]))
    transaction.set <- transaction.set[transaction.set != ""]
    dataset <- list.append(dataset, transaction.set)
  }
  return(dataset)
}


##################### 2단계 : datasets change to dataframe

get_item_freq_table <- function(dataset){
  item.freq.table <- unlist(dataset) %>% table %>% data.frame
  return(item.freq.table)
}


##################### 3단계 : Function to cut items based on the minimum frequency
# Defined by the user, here the minimum frequency is specified as item.min.freq

prune_item_freq_table <- function(item.freq.table, item.min.freq){
  pruned.item.table <- item.freq.table[item.freq.table$Freq >= item.min.freq,]
  return(pruned.item.table)
}



##################### 4 단계 : A function to get an item set with n items
#  n means num.item

get_associated_itemset_combinations <- function(pruned.item.table, num.items){
  itemset.associations <- c()
  itemset.association.matrix <- combn(pruned.item.table$.,num.items)
  for (index in seq(ncol(itemset.associations.matrix))){
    itemset.associations <- c(itemset.associations, paste(itemset.association.matrix[,index], collapse = ", ")
    )
  }
  itemset.associations <- unique(itemset.associations)
  return(itemset.associations)
}


###################### 5단계 : A function to create an item set association matrix

build_itemset_association_matrix <- function(dataset, itemset.association.labels, itemset.combination.nums){
  itemset.transaction.labels <- sapply(dataset, paste, collapse=", ")
  itemset.associations <- lapply(itemset.association.labels, 
                                 function(itemset){
                                   unlist(strsplit(itemset, ", ", 
                                                   fixed=TRUE)
                                   )
                                 }
  )
  # Creating an item set association matrix
  association.vector <- c()
  for (itemset.association in itemset.associations){
    association.vector <- c(association.vector,
                            unlist(
                              lapply(dataset,
                                     function(dataitem,
                                              num.items=itemset.combination.nums){
                                       m <- match(dataitem, itemset.association)
                                       m <- length(m[!is.na(m)])
                                       if ( m == num.items){
                                         1
                                       } else {
                                         NA
                                       }
                                       
                                     }
                              )
                            )
    )
  }
  
  itemset.association.matrix <- matrix(association.vector, 
                                       nrow = length(dataset))
  itemset.association.labels <- sapply(itemset.association.labels,
                                       function(item) {
                                         paste0('{',paste(item,collapse = ", "),"}")
                                       }
  )
  itemset.transaction.labels <- sapply(dataset, 
                                       function(itemset){
                                         paste0('{', paste(itemset,
                                                           collapse = ', '), "}")
                                       }
  )
  colnames(itemset.association.matrix) <- itemset.association.labels
  rownames(itemset.association.matrix) <- itemset.transaction.labels
  
  return (itemset.association.matrix)
}


###################### 6단계 : 연관 행렬로부터 각 아이템 세트 발생 횟수의 총합을 구하는 함수

get_frequent_itemset_details <- function(itemset.association.matrix){
  frequent.itemsets.table <- apply(itemset.association.matrix,2, sum, na.rm=TRUE)
  return(frequent.itemsets.table)
}

#########################7단계 : A function that calculates the sum of the number of occurrences of each item set from the association matrix

frequent.itemsets.generator <- function(data.file.path, itemset.combination.nums=2, item.min.freq=2, minsup=0.2){
  # datasets
  dataset <- get_transaction_dataset(data.file.path)
  
  # Make data into item frequency table
  item.freq.table <- get_item_freq_table(dataset)
  pruned.item.table <- prune_item_freq_table(item.freq.table, item.min.freq)
  
  # Getting Item Set Association
  itemset.association.labels <- get_associated_itemset_combinations(pruned.item.table, itemset.combination.nums)
  itemset.association.matrix <- build_itemset_association_matrix(dataset, itemset.association.labels, itemset.combination.nums)
  
  # Create a set of frequent items
  frequent.itemsets.table <- get_frequent_itemset_details(itemset.association.matrix)
  frequent.itemsets.table <- sort(frequent.itemsets.table[frequent.itemsets.table > 0], decreasing = TRUE)
  
  frequent.itemsets.names <- names(frequent.itemsets.table)
  frequent.itemsets.frequencies <- as.vector(frequent.itemsets.table)
  frequent.itemsets.support <- round((frequent.itemsets.frequencies * 100)/length(dataset), digits = 2)
  frequent.itemsets <- data.frame(Itemset=frequent.itemsets.names, Frequency = frequent.itemsets.frequencies, Support=frequent.itemsets.support)
  
  # Cut to the minimum support value from the obtained set of frequent items
  minsup.percentage <- minsup * 100
  frequent.itemsets <- subset(frequent.itemsets, frequent.itemsets['Support']>=minsup.percentage)
  frequent.itemsets.support <- sapply(frequent.itemsets.support, function(value){
    paste0(value,'%')
  }
  )
  
  # print to console
  cat("\nItem Association Matrix\n")
  print(itemset.association.matrix)
  cat("\n\n")
  cat("\nVaild Frequent Itemsets with Frequency and Support\n")
  print(frequent.itemsets)
  
  # print to table
  if (names(dev.cur()) !="null device"){
    dev.off()
  }
  grid.table(frequent.itemsets)
}

# find shopping trend
# Consists of 2 items, at least 2 purchases, minimum approval rating of 20%
frequent.itemsets.generator(
  data.file.path = 'shopping_transaction_log.csv',itemset.combination.nums =2, item.min.freq=2, minsup=0.2)

# Consists of 3 items, at least 1 purchases, minimum approval rating of 20%
frequent.itemsets.generator(
  data.file.path = 'shopping_transaction_log.csv',itemset.combination.nums =3, item.min.freq=1, minsup=0.2)



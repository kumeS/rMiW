
lr_schedule_10 <- function(epoch, lr) {
  if(epoch <= 10) {
    0.01
  } else if(epoch > 10 && epoch <= 20){
    0.001
  } else if(epoch > 20 && epoch <= 30){
    0.0001
  } else {
    0.00001 }}

lr_schedule_25 <- function(epoch, lr) {
  if(epoch <= 25) {
    0.001
  } else if(epoch > 25 && epoch <= 50){
    0.0001
  } else if(epoch > 50 && epoch <= 75){
    0.00001
  } else {
    0.00001 }}

lr_schedule_100 <- function(epoch, lr) {
  if(epoch <= 100) {
    0.01
  } else if(epoch > 100 && epoch <= 200){
    0.001
  } else if(epoch > 200 && epoch <= 300){
    0.0001
  } else {
    0.0001 }}

lr_schedule_200 <- function(epoch, lr) {
  if(epoch <= 200) {
    0.01
  } else if(epoch > 200 && epoch <= 400){
    0.001
  } else if(epoch > 400 && epoch <= 600){
    0.0001
  } else {
    0.0001 }}

lr_schedule_500 <- function(epoch, lr) {
  if(epoch <= 500) {
    0.01
  } else if(epoch > 500 && epoch <= 1000){
    0.001
  } else if(epoch > 1000 && epoch <= 1500){
    0.0001
  } else {
    0.0001 }}

lr_schedule_1000 <- function(epoch, lr) {
  if(epoch <= 1000) {
    0.01
  } else if(epoch > 1000 && epoch <= 2000){
    0.001
  } else if(epoch > 2000 && epoch <= 3000){
    0.0001
  } else {
    0.0001 }}


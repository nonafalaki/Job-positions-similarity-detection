from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
torch.cuda.empty_cache()
def main():
  jobtitle1=input('What is the job title1? ')
  jobtitle2=input('What is the job title2? ')
  #set the pretrained model
  #'stsb-roberta-large' model performs better but requires more GPU space	
  model = SentenceTransformer('stsb-roberta-base')		
  sentence_embeddings=model.encode([jobtitle1,jobtitle2])
  #check for similarity
  similarity=cosine_similarity([sentence_embeddings[0]],[sentence_embeddings[1]])
  #set threshold to 50
  if similarity[0,0] >=0.50:
    print(jobtitle1,jobtitle2,'are',similarity,'similar')
  else:
    print(jobtitle1,'and',jobtitle2,'are not similar')
  status=input('Do you wanna try more?')  
  if status.lower()=='yes':
    main()
  elif status.lower()!='yes' and status.lower()!='no':
    print('please insert yes or no!')
main()

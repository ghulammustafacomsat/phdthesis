from IPython.core.display import publish_display_data
import csv
import numpy as np
import pandas as pd
import statistics
import numpy as np
import math
from datetime import date
list_l=[]
kk=0
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

def publication_count(publication_id):
  return len(publication_id)

def citation_count(citation):
  return (sum(citation))

def total_years(publication_years):
  return (max(publication_years)-min(publication_years))

def cite_per_years(citation,years):
  t_y=(max(years)-min(years))
  if(t_y==0):
    result=(sum(citation))/1
  else:
    result =(sum(citation))/t_y
  return result

def cite_per_paper(citation,publication):
  return ((sum(citation))/(len(publication)))
  
  
def Author_paper(publication,authors):
  author_p=0
  author=0
  paper=0
  author=sum(authors)
  paper=len(publication)
  author_p=author/paper
  return author_p

def cites_author(citation, publication,authors):
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=citation[i]/authors[i]
    c_ss.append(c_s) 
  return sum(c_ss)

def papers_author(publication,authors):
  publication.sort()
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=1/authors[i]
    c_ss.append(c_s) 
  return sum(c_ss)

def h_index_f(citation,publication):
  h_index=0
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      h_index=1
    elif (len(publication)==1 and citation[i]==0):
      h_index=0
    elif(publication[i]<=citation[i]):
      h_index=publication[i]
  return h_index

def g_index_f(citation,publication):
  g_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  c_sum=[]
  pub_s=[i ** 2 for i in publication]
  c_sum=Cumulative(citation)
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      g_index=1
    elif (len(publication)==1 and citation[i]==0):
      g_index=0
    elif(pub_s[i]<=c_sum[i]):
      g_index=publication[i]
  return g_index

def h2_index_f(citation,publication):
  h2_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  pub_s=[i ** 2 for i in publication]
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      h2_index=1
    elif (len(publication)==1 and citation[i]==0):
      h2_index=0
    elif(pub_s[i]<=citation[i]):
      h2_index=publication[i]
  return h2_index

def w_index_f(citation,publication):
  w_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  pub_s=[i * 10 for i in publication]
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=10):
      w_index=1
    elif (len(publication)==1 and citation[i]<10):
      w_index=0
    elif(pub_s[i]<=citation[i]):
      w_index=publication[i]
  return w_index


def f_index(citation, publication):
  f_index=0
  c_list=[]
  s_list=[]
  f_list=[]
  r_list=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(citation[i]==0):
      c_list.append(0)
    else:
      c_list.append(1/citation[i])
  for j in range(len(publication)+1):
    summ=0
    for x in range(j):
      summ=summ+c_list[x]
    if(j==0):
      continue
    s_list.append(summ)
  for j in range(len(publication)):
    f_list.append(s_list[j]/publication[j])
  for k in range(len(publication)):
    if(f_list[k]==0):
      r_list.append(0)
    else:  
      r_list.append(1/f_list[k])
  for n in range(len(publication)):
    if(publication[n]>r_list[n]):
      f_index=publication[n]-1
      break
  if(f_index==0):
    f_index=len(publication)
  return f_index

def t_index(citation, publication):
  t_index=0
  l_list=[]
  lt_list=[]
  lm_list=[]
  ln_list=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(citation[i]==0):
      l_list.append(0.0)
    else:
      l_list.append(np.log(citation[i]))   
  for j in range(len(publication)+1):
    summ=0
    for x in range(j):
      summ=summ+l_list[x]
    if(j==0):
      continue
    lt_list.append(summ)
  for n in range(len(publication)):
    lm_list.append(lt_list[n]/publication[n])
  for k in range(len(publication)):
      ln_list.append(np.exp(lm_list[k]))
  for m in range(len(publication)):
    if(publication[m]>ln_list[m]):
      t_index=publication[m]-1
      break
  if(t_index==0):
    t_index=len(publication)
  return t_index

def woginger_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  wo_index=0
  for j in range(len(publication)):
    if(citation[j]>=(len(publication)-publication[j]+1)):
      wo_index=wo_index+1
  return wo_index

def h_core_citation(citation,publication):
  h_core=0
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(publication[i]<citation[i]):
      h_core=h_core+citation[i]
  return h_core

def m_index(citation, publication):
  listt=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(publication[i]<citation[i]):
      listt.append(citation[i])
  if(len(listt)==0):
    return 0
  else:
    return statistics.median(listt)

def tappered_h_index(citation,publication):
    th_list=[]
    citation.sort(reverse=True)
    publication.sort()
    for i in range(len(publication)):
      a=0 
      b=0
      c=0
      r=0
      if(citation[i]<=publication[i]):
        a=citation[i]/(2*publication[i]-1)
        th_list.append(a)
      else:
        b=publication[i]/(2*publication[i]-1)
        r=publication[i]+1
        for r in range(int(citation[i])):
          c=1/(2*publication[i]-1)
        th_list.append(b+c)
    return sum(th_list) 

def Maxprod_index(citation, publication):
    Mi_list=[]
    citation.sort(reverse=True)
    publication.sort()
    for i in range(len(publication)):
      Mi_list.append(citation[i]*publication[i])        
    return max(Mi_list) 

def wu_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  wu_list=[]
  wu1_list=[]
  wu_i=0
  for i in range(len(publication)):
    wu_list.append(10*citation[i])
    wu1_list.append(10*publication[i])
  for j in range(len(publication)):
    if(publication[j]<=wu_list[j]):
      wu_i=j+1
  return wu_i  

def pi_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  pi_index=0
  sq=0
  sum_c=0
  sq=math.sqrt(len(publication))
  for j in range(int(sq)):
    sum_c=sum_c+citation[j]
  pi_index=0.01*sum_c
  return pi_index

def weighted_h_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  hw_index=0
  m=0
  wl_list=[]
  for i in range(len(publication)):
    if(publication[i]<=citation[i]):
      h_index=publication[i]
  for j in range(len(publication)+1):
    sum=0
    for p in range(j):
      sum=sum+citation[p]
    if(h_index==0):
      wl_list.append(0)
    else:
      wl_list.append(sum/h_index)
  wl_list.pop(0)
  for k in range(len(publication)):
    if(wl_list[k]<=citation[k]):
      m=k+1
  for r in range(m):
    hw_index=hw_index+citation[r]
  return math.sqrt(hw_index)


def Gh_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  gh_list=[]
  gs_list=[]
  gh_index=0
  h_index=h_index_f(citation,publication)
  for j in range(len(publication)):
    gh_list.append(citation[j]-h_index)
  for k in range(len(publication)):
    if(gh_list[k]>=0):
      gh_index=gh_index+1
  return gh_index

def A_index(citation, publication):
  hcc=0
  h_index=0
  a_index=0
  citation.sort(reverse=True)
  publication.sort()
  hcc=h_core_citation(citation,publication)
  h_index=h_index_f(citation,publication)
  if(h_index==0):
    a_index=0
  else:
    a_index=hcc/h_index
  return a_index

def R_index(citation, publication):
  hcc=0
  R_index=0
  citation.sort(reverse=True)
  publication.sort()
  hcc=h_core_citation(citation,publication)
  R_index=math.sqrt(hcc)
  return R_index

def Rm_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  rm_index=0
  h_index=0
  rm_list=[]
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    rm_list.append(math.sqrt(citation[j]))
  rm_index=math.sqrt(sum(rm_list))
  return rm_index

def x_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  x_list=[]
  ma=0
  sq=0.0
  x_index=0
  for i in range(len(publication)):
      x_list.append(publication[i]*citation[i])
  ma=max(x_list)
  x_index=x_list.index(ma)
  sq=math.sqrt(x_index+1)
  return sq

def h2upper_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h2list=[]
  sum1=0
  sum2=0
  h2upper=0
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    h2list.append(citation[j]-h_index)
  sum1=sum(h2list)
  sum2=sum(citation)
  if(sum2==0):
    h2upper=0
  else:
    h2upper=(sum1/sum2)*100
  return h2upper

def h2center_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  sum1=0
  h2center=0
  h_index=h_index_f(citation,publication)
  sum1=sum(citation)
  if(sum1==0):
    h2center=0
  else:
    h2center=(h_index*h_index/sum1)*100
  return h2center

def h2lower_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h2llist=[]
  sum1=0
  sum2=0
  h2lower=0
  h_index=h_index_f(citation,publication)
  km=h_index+1
  for j in range(km,len(publication)):
    h2llist.append(citation[j]-h_index)
  sum1=sum(h2llist)
  sum2=sum(citation)
  if(sum2==0):
    h2lower=0
  else:
    h2lower=(sum1/sum2)*100
  return h2lower

def k_dash_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  sumh=0
  sumt=0
  k_dash_index=0
  citall=sum(citation)
  pubcount=len(publication)
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    sumh=sumh+citation[j]
  km=h_index+1
  for k in range(km,len(publication)):
    sumt=sumt+citation[k]
  if(sumt==0 or sumh==0 or (sumt-sumh)==0):
    k_dash_index=0
  else:
    k_dash_index=(citall-pubcount)/(sumt-sumh)
  return k_dash_index


def iten_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  count=0
  for i in range(len(publication)):
    if(citation[i]>=10):
      count=count+1
  return count

def normalized_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  nh_index=0
  h_index=0
  pub=0
  pub=len(publication)
  h_index=h_index_f(citation,publication)
  nh_index=h_index/pub  
  return nh_index

def platinium_h_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  ph_index=0
  CL=0
  total_cit=0
  total_pub=0
  CL=max(years)-min(years)
  total_cit=sum(citation)
  total_pub=len(publication)
  h_index=h_index_f(citation,publication)
  if(CL==0):
    ph_index=(h_index)*(total_cit/total_pub)
  else:
    ph_index=(h_index/CL)*(total_cit/total_pub)
  return ph_index

def m_qoutient_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  CL=0
  CL=max(years)-min(years)
  m_quotient=0
  h_index=h_index_f(citation,publication)
  if(CL==0):
    m_quotient=(h_index)
  else:
    m_quotient=(h_index/CL)
  return m_quotient

def HI_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  HI_index=0
  listt=[]
  h_index=h_index_f(citation,publication)
  for j in range(len(publication)):
    listt.append(authors_l[j])
  average_au=sum(listt)/len(listt)
  HI_index=h_index/average_au
  return HI_index

def AW_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  l1=[]
  l2=[]
  summ=0
  papers=1
  for j in range(len(years)):
    summ=0
    papers=0
    for k in range(len(years)):
      if(years[j]==years[k]):
        summ=summ+citation[k]
        papers=papers+1
    l1.append(years[j])
    l2.append(summ/papers)
  return math.sqrt(sum(l2))

def Ar_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  AW_index=0
  h_core_p=[]
  h_core_c=[]
  h_core_y=[]
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    h_core_p.append(publication[j])
  for c in range(h_index):
    h_core_c.append(citation[c])   
  for y in range(h_index):
    h_core_y.append(years[y])
  cleanedList = [x for x in set(h_core_y) if str(x) != 'nan']
  l1=[]
  l2=[]
  summ=0
  papers=1
  for j in range(len(cleanedList)):
    summ=0
    papers=0
    for k in range(len(h_core_y)):
      if(cleanedList[j]==h_core_y[k]):
        summ=summ+h_core_c[k]
        papers=papers+1
    l1.append(cleanedList[j])
    l2.append(summ/papers)
  return math.sqrt(sum(l2))

def v_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  import datetime
  today = datetime.date.today()
  cyear = today.year
  CL=0
  CL=cyear-min(years)
  v_index=0
  for i in range(len(publication)):
    if(publication[i]<=citation[i]):
      h_index=publication[i]
  v_index=(h_index/CL)
  return v_index

def hm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  hm_index=0
  weights=[]
  for k in range(len(publication)):
    if (authors_l[k]!=0):
      weights.append(1/authors_l[k])
    else:
      weights.append(1)
  new_list=[]
  j=0
  for i in range(0,len(weights)):
      j+=weights[i]
      new_list.append(j)  
  for m in range(len(publication)):
    if(new_list[m]<=citation[m]):
      hm_index=new_list[m]
  return hm_index

def gm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  eff_rank=0
  ef_rank=[]
  normalized_c=[]
  citation_ss=0
  citation_s=[]
  effe_cit=[]
  gm_index=0
  flag=0
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for m in range(len(publication)):
    effe_cit.append(citation_s[m]/ef_rank[m])
  for l in range(len(publication)):
    if(ef_rank[l]>citation_s[l]):
      gm_index=ef_rank[l-1]
      flag=1
      break
  if(flag==0): 
    gm_index=len(publication)
  return gm_index
  

def hf_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  flag=0
  hf_index=0
  normalized_c=[]
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for k in range(len(publication)):
    if(publication[k]>normalized_c[k]):
      hf_index=k+1
      flag=1
      break
  if(flag==0):
    hf_index=len(publication) 
  return hf_index

def gf_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  gf_index=0
  normalized_c=[]
  flag=0
  rank_s=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for k in range(len(publication)):
    rank_s.append(publication[k]*publication[k])
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(rank_s[l]>=citation_s[l]):
      gf_index=publication[l]
      flag=1
      break
  if(flag==0): 
    gf_index=len(publication)
  return gf_index

def gF_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  gF_index=0
  flag=0
  citation_s=[]
  citation_ss=0
  ef_rank=[]
  eff_rank=0
  eff_rank_square=[]
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for m in range(len(publication)):
    eff_rank_square.append(ef_rank[m]*ef_rank[m])
  for l in range(len(publication)):
    if(eff_rank_square[l]>=citation_s[l]):
      gF_index=publication[l]
      break
  if(flag==0): 
    gF_index=len(publication)
  return gF_index

def hi_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_i_index=0
  coauthor=[]
  coauthors=0
  h_index=h_index_f(citation,publication)
  for k in range(h_index):
    coauthors=coauthors+authors_l[k];
  if(coauthors==0):
    h_i_index=(h_index*h_index);
  else:
    h_i_index=(h_index*h_index)/coauthors;
  return h_i_index

def h_m_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  i=0
  hm_index=0
  coauthor=[]
  flag=0
  less=0
  ef_rank=[]
  eff_rank=0
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for l in range(len(publication)):
    if(ef_rank[l]>citation[l]):
      hm_index=ef_rank[l]
      break
    elif (ef_rank[l]==citation[l]):
      flag=flag+1
    elif(ef_rank[l]<citation[l]):
      less=less+1
  hm_index=ef_rank[(flag+less)-1]
  return hm_index

def k_norm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  k_norm=0
  h_norm=0
  equal=0
  greater=0
  coauthor=[]
  normalized_c=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(normalized_c[l]<publication[l]):
      h_norm=l
      break
    elif(normalized_c[l]==publication[l]):
      equal=equal+1
    elif(normalized_c[l]>publication[l]):
      greater=greater+1
  h_norm=publication[(greater+equal)-1]
  if(citation_s[h_norm-1]==0):
    k_norm=0
  else:
    k_norm=h_norm+(1-(h_norm*h_norm)/citation_s[h_norm-1])
  return k_norm

def w_norm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  w_norm=0
  h_norm=0
  equal=0
  greater=0
  coauthor=[]
  normalized_c=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(normalized_c[l]<publication[l]):
      h_norm=l
      break
    elif(normalized_c[l]==publication[l]):
      equal=equal+1
    elif(normalized_c[l]>publication[l]):
      greater=greater+1
  h_norm=publication[(greater+equal)-1]
  if(citation_s[len(citation_s)-1]<=0):
    w_norm=0
  else:
    w_norm=h_norm+(1-(h_norm*h_norm)/citation_s[len(citation_s)-1])
  return w_norm

def A_index(citation, publication):
  hcc=0
  h_index=0
  a_index=0
  citation.sort(reverse=True)
  publication.sort()
  hcc=h_core_citation(citation,publication)
  h_index=h_index_f(citation,publication)
  if(h_index==0):
    a_index=0
  else:
    a_index=hcc/h_index
  return a_index



def P_index(citation, publication):
  sc=0
  p_index=0
  citation.sort(reverse=True)
  publication.sort()
  sc=sum(citation)
  p=len(publication)
  p_index=sc/p
  return p_index

def q_sq_index(citation, publication):
  h_index=0
  m_index1=0
  q_sq_index1=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  m_index1=m_index(citation,publication)
  q_sq_index1=math.sqrt(h_index*m_index1)
  return q_sq_index1

def e_index(citation, publication):
  h_index=0
  e_index=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  hcc=h_core_citation(citation,publication)
  e_index=hcc-(h_index*h_index)
  return e_index

def hg_index(citation, publication):
  h_index=0
  g_index=0
  hg_index=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  g_index=g_index_f(citation,pub_id)
  hg_index=math.sqrt(h_index*g_index)
  return hg_index

def pure_h_index(citation, publication,author):
  h_index=0
  author_c=0
  author_av=0
  pure_h_indexx=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  for i in range(h_index):
    author_c=author_c+author[i]
  if(h_index==0):
    author_av=0
  else:
    author_av=author_c/h_index
  if(author_av==0):
    pure_h_indexx=0
  else:
    pure_h_indexx=h_index/math.sqrt(author_av)
  return pure_h_indexx

def fractional_g_index_f(citation,publication,author):
  frac_g_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  c_su=0
  c_sum=[]
  pub_s=[i ** 2 for i in publication]
  for i in range(len(publication)):
    c_su=c_su+citation[i]/author[i]
    c_sum.append(c_su)  
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      frac_g_index=1
    elif (len(publication)==1 and citation[i]==0):
      frac_g_index=0
    elif(pub_s[i]<=c_sum[i]):
      frac_g_index=publication[i]
  return frac_g_index

def fractional_h_index_f(citation,publication,author):
  frac_h_index=0
  citation.sort(reverse=True)
  publication.sort()
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=citation[i]/author[i]
    c_ss.append(c_s) 
  for j in range(len(publication)):
    if (len(publication)==1 and c_ss[j]>=1):
      frac_h_index=1
    elif (len(publication)==1 and c_ss[j]==0):
      frac_h_index=0
    elif(publication[j]<=c_ss[j]):
      frac_h_index=publication[j]
  return frac_h_index

def AWCR_index_f(citation,publication,year):
  awcr_index=0
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_d=0
  c_s=0
  sum=0
  for i in range(len(publication)):
    c_d=cyear-year[i]
    if(c_d!=0):
      c_s=citation[i]/c_d
    else:
      c_s=citation[i]/1
    awcr_index=awcr_index+c_s
  return awcr_index



def k_index(citation,publication,authors):
  citation.sort(reverse=True)
  publication.sort()
  k_index=0
  c_t=0
  p_t=0
  h_index=0
  c_total=sum(citation)
  p_total=len(publication)
  h_index=h_index_f(citation,publication)
  c_s=0
  c_core=0
  c_t=0
  c_tail=0
  for i in range(h_index):
    c_s=c_s+citation[i]
  c_core=c_s
  for i in range(h_index, len(publication)):
    c_t=c_t+citation[i]
  c_tail=c_t
  if(c_tail==0):
    if(c_core==0):
      k_index=0
    else:
      k_index=(c_total/p_total)/c_core
  else:
    k_index=(c_total/p_total)/(c_tail/c_core)
  return k_index

def h_dash_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_dash_indexx=0
  e_indexx=0
  t_indexx=0
  h_indexx=0
  e_indexx=e_index(citation, publication)
  t_indexx=t_index(citation, publication)
  h_indexx=h_index_f(citation,publication)
  h_dash_indexx=(e_indexx/t_indexx)*h_indexx
  return h_dash_indexx

def hl_norm(citation,publication,author):
  citation.sort(reverse=True)
  publication.sort()
  c_t=[]
  h_norm=0
  for i in range(len(publication)):
    c_t.append(citation[i]/author[i])
  c_t.sort(reverse=True)
  h_norm=h_index_f(c_t,publication)
  return h_norm

def hc_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_t=[]
  sc=0
  hc_index=0
  for i in range(len(publication)):
    sc=4*(citation[i]/(cyear-years[i]+1))
    c_t.append(sc)
  c_t.sort(reverse=True)
  hc_index=h_index_f(c_t,publication)
  return hc_index

def ha_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_t=[]
  sc=0
  ha_indexx=0
  for i in range(len(publication)):
    if(cyear-years[i]!=0):
      sc=citation[i]/(cyear-years[i])
    else:
      sc=citation[i]
    c_t.append(sc)
  c_t.sort(reverse=True)
  ha_indexx=h_index_f(c_t,publication)
  return ha_indexx

def hi_index_s(citation,publication,authors):
  citation.sort(reverse=True)
  publication.sort()
  h_indexx=0
  h_indexx=h_index_f(citation,publication)
  hi_index_ss=0
  c_t=0
  c=0
  sc=0
  ha_indexx=0
  for i in range(h_indexx):
    c=c+1
    sc=sc+authors[i]
  if(c==0):
    c_t=0
    hi_index_ss=0
  else:
    c_t=sc/c
    hi_index_ss=h_indexx/c_t
  return hi_index_ss

def rational_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_index=h_index_f(citation,publication)
  k=0
  h_r_index=0
  if(h_index==len(publication)):
    k=citation[h_index-1]
    h_r_index=(h_index+1)-k/(2*h_index+1)
  else:
    k=citation[h_index]-citation[h_index-1]
    h_r_index=(h_index+1)-k/(2*h_index+1)
  return h_r_index

def real_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_index=h_index_f(citation,publication)
  k=0
  z=0
  r_h_index=0
  if(h_index==len(publication)):
    k=(h_index+1)* citation[h_index-1] - h_index*citation[h_index-1]+1 
    z=1-citation[h_index-1]
  else:
    k=(h_index+1)* citation[h_index-1] - h_index*citation[h_index-1]+1 
    z=1-citation[h_index] + citation[h_index-1]
  if(z==0):
    r_h_index= 0
  else:
    r_h_index= k/z
  return r_h_index


def Author_indexs_c(pub_id,citation,author_name,publication_y,co_auth):
  row=[]
  row.append(author_name)
  row.append(publication_count(pub_id))
  row.append(citation_count(citation))
  row.append(total_years(publication_y))
  row.append(cite_per_years(citation,publication_y))
  row.append(cite_per_paper(citation,pub_id))
  row.append(h_index_f(citation,pub_id))
  row.append(g_index_f(citation,pub_id))
  row.append(h2_index_f(citation,pub_id))
  row.append(w_index_f(citation,pub_id))
  row.append(h_core_citation(citation,pub_id))
  row.append(m_index(citation,pub_id))
  row.append(f_index(citation,pub_id))
  row.append(t_index(citation,pub_id))
  row.append(tappered_h_index(citation,pub_id))
  row.append(Maxprod_index(citation,pub_id))
  row.append(wu_index(citation,pub_id))
  row.append(pi_index(citation,pub_id))
  row.append(weighted_h_index(citation,pub_id))
  row.append(woginger_index(citation,pub_id))
  row.append(Gh_index(citation,pub_id))
  row.append(Rm_index(citation,pub_id))
  row.append(x_index(citation,pub_id))
  row.append(h2upper_index(citation,pub_id))
  row.append(h2center_index(citation,pub_id))
  row.append(h2lower_index(citation,pub_id))
  row.append(k_dash_index(citation,pub_id))
  row.append(iten_index(citation,pub_id))
  row.append(normalized_h_index(citation,pub_id))
  row.append(platinium_h_index(citation,pub_id,publication_y))
  row.append(m_qoutient_index(citation,pub_id,publication_y))
  row.append(HI_index(citation,pub_id,co_auth))
  row.append(AW_index(citation,pub_id,publication_y))
  row.append(Ar_index(citation,pub_id,publication_y))
  row.append(v_index(citation,pub_id,publication_y))
  row.append(hm_index(citation,pub_id,co_auth))
  row.append(hf_index(citation,pub_id,co_auth))
  row.append(gf_index(citation,pub_id,co_auth))
  row.append(gF_index(citation,pub_id,co_auth))
  row.append(hi_index(citation,pub_id,co_auth))
  row.append(h_m_index(citation,pub_id,co_auth))
  row.append(k_norm_index(citation,pub_id,co_auth))
  row.append(w_norm_index(citation,pub_id,co_auth))
  row.append(gm_index(citation,pub_id,co_auth))
  row.append(A_index(citation,pub_id))
  row.append(R_index(citation,pub_id))
  row.append(P_index(citation,pub_id))
  row.append(q_sq_index(citation,pub_id))
  row.append(e_index(citation,pub_id))
  row.append(hg_index(citation,pub_id))
  row.append(pure_h_index(citation,pub_id,co_auth))
  row.append(fractional_g_index_f(citation,pub_id,co_auth))
  row.append(fractional_h_index_f(citation,pub_id,co_auth))
  row.append(AWCR_index_f(citation,pub_id,publication_y))
  row.append(Author_paper(pub_id,co_auth))
  row.append(cites_author(citation, pub_id,co_auth))
  row.append(papers_author(pub_id,co_auth))
  row.append(k_index(citation, pub_id,co_auth))
  row.append(h_dash_index(citation, pub_id))
  row.append(hl_norm(citation,pub_id,co_auth))
  row.append(hc_index(citation,pub_id,publication_y))
  row.append(ha_index(citation,pub_id,publication_y))
  row.append(hi_index_s(citation,pub_id,co_auth))
  row.append(rational_h_index(citation,pub_id))
  row.append(real_h_index(citation,pub_id))
  print(row)
  list_l.append(row)
  


data = pd.read_csv('UpdateAwardeesmerge.csv',encoding='latin-1')
authors_name=[]
publication=[]
citation=[]
publishing_years=[]
co_author=[]
for col in data.itertuples(): #reading file
  authors_name.append(col[1])
  publication.append(col[2])
  citation.append(col[3])
  publishing_years.append(col[4])
  co_author.append(col[5])
author_name=''
pub_id=[]
citation_c=[]
publication_y=[]
co_auth=[]
dff = pd.DataFrame(authors_name, columns=['author_name'])
gk=dff.groupby('author_name').agg(lambda x: ','.join(x))
author_l=gk.index.tolist()
k=0
for j in range(len(author_l)):
  for r in range(len(authors_name)):
    author_name=author_l[j]
    if(author_l[j]==authors_name[r]):
      pub_id.append(publication[r])
      citation_c.append(citation[r])
      publication_y.append(publishing_years[r])
      co_auth.append(co_author[r])
    else:
      continue
  k=k+1
  print(k , author_name)
  citation_c=[int(a) for a in citation_c]
  publication_y=[int(a) for a in publication_y]
  co_auth=[int(a) for a in co_auth]
  print(citation_c)
  Author_indexs_c(pub_id,citation_c,author_name,publication_y,co_auth)
  pub_id=[]
  citation_c=[]
  publication_y=[]
  co_auth=[]

file = open('UpdateAwardeesmerge_file.csv', 'w+', newline ='')
header = [['Author Name', 'Total Publication', 'Total Citation','Total Years','Cite/Year','Cite/Paper','h index','g index',
           'h(2) index', 'w index','h core citation', 'm index', 'f index', 't index', 'tappered_h_index', 'Maxprod',
           'wu index', 'pi index', 'weighted h index','woginger_index', 'Gh_index', 'Rm index', 'X index','h2upper_index',
           'h2 center_index','h2 lower_index','k dash index','i10 index','normalized_h_index','platinium_h_index',
           'm_qoutient_index','HI_index','AW_index','Ar index','V index','hm_index','hf index','gf index','gF index',
           'hi index','h_m_index','k norm index','w norm index','gm index','A index','R index','P index','q2 index',
           'e index', 'hg_index','pure h index','fractional g index','fractional h index','AWCR','Author/paper',
           'cites/author','paper/author','k index','h dash index','hl norm','hc index','ha index','hi index s',
           'Rational h index', 'Real h index']]
with file:   
  write = csv.writer(file)
  write.writerows(header)
  write.writerows(list_l)

from IPython.core.display import publish_display_data
import csv
import numpy as np
import pandas as pd
import statistics
import numpy as np
import math
from datetime import date
list_l=[]
kk=0
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

def publication_count(publication_id):
  return len(publication_id)

def citation_count(citation):
  return (sum(citation))

def total_years(publication_years):
  return (max(publication_years)-min(publication_years))

def cite_per_years(citation,years):
  t_y=(max(years)-min(years))
  if(t_y==0):
    result=(sum(citation))/1
  else:
    result =(sum(citation))/t_y
  return result

def cite_per_paper(citation,publication):
  return ((sum(citation))/(sum(publication)))

def h_index_f(citation,publication):
  h_index=0
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      h_index=1
    elif (len(publication)==1 and citation[i]==0):
      h_index=0
    elif(publication[i]<=citation[i]):
      h_index=publication[i]
  return h_index

def g_index_f(citation,publication):
  g_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  c_sum=[]
  pub_s=[i ** 2 for i in publication]
  c_sum=Cumulative(citation)
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      g_index=1
    elif (len(publication)==1 and citation[i]==0):
      g_index=0
    elif(pub_s[i]<=c_sum[i]):
      g_index=publication[i]
  return g_index

def h2_index_f(citation,publication):
  h2_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  pub_s=[i ** 2 for i in publication]
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      h2_index=1
    elif (len(publication)==1 and citation[i]==0):
      h2_index=0
    elif(pub_s[i]<=citation[i]):
      h2_index=publication[i]
  return h2_index

def w_index_f(citation,publication):
  w_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  pub_s=[i * 10 for i in publication]
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=10):
      w_index=1
    elif (len(publication)==1 and citation[i]<10):
      w_index=0
    elif(pub_s[i]<=citation[i]):
      w_index=publication[i]
  return w_index

def h_core_citation(citation,publication):
  h_core=0
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(publication[i]<citation[i]):
      h_core=h_core+citation[i]
  return h_core

def m_index(citation, publication):
  listt=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(publication[i]<citation[i]):
      listt.append(citation[i])
  if(len(listt)==0):
    return 0
  else:
    return statistics.median(listt)

def f_index(citation, publication):
  f_index=0
  c_list=[]
  s_list=[]
  f_list=[]
  r_list=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(citation[i]==0):
      c_list.append(0)
    else:
      c_list.append(1/citation[i])
  for j in range(len(publication)+1):
    summ=0
    for x in range(j):
      summ=summ+c_list[x]
    if(j==0):
      continue
    s_list.append(summ)
  for j in range(len(publication)):
    f_list.append(s_list[j]/publication[j])
  for k in range(len(publication)):
    if(f_list[k]==0):
      r_list.append(0)
    else:  
      r_list.append(1/f_list[k])
  for n in range(len(publication)):
    if(publication[n]>r_list[n]):
      f_index=publication[n]-1
      break
  if(f_index==0):
    f_index=len(publication)
  return f_index

def t_index(citation, publication):
  t_index=0
  l_list=[]
  lt_list=[]
  lm_list=[]
  ln_list=[]
  citation.sort(reverse=True)
  publication.sort()
  for i in range(len(publication)):
    if(citation[i]==0):
      l_list.append(0.0)
    else:
      l_list.append(np.log(citation[i]))   
  for j in range(len(publication)+1):
    summ=0
    for x in range(j):
      summ=summ+l_list[x]
    if(j==0):
      continue
    lt_list.append(summ)
  for n in range(len(publication)):
    lm_list.append(lt_list[n]/publication[n])
  for k in range(len(publication)):
      ln_list.append(np.exp(lm_list[k]))
  for m in range(len(publication)):
    if(publication[m]>ln_list[m]):
      t_index=publication[m]-1
      break
  if(t_index==0):
    t_index=len(publication)
  return t_index

def tappered_h_index(citation,publication):
    th_list=[]
    citation.sort(reverse=True)
    publication.sort()
    for i in range(len(publication)):
      a=0 
      b=0
      c=0
      r=0
      if(citation[i]<=publication[i]):
        a=citation[i]/(2*publication[i]-1)
        th_list.append(a)
      else:
        b=publication[i]/(2*publication[i]-1)
        r=publication[i]+1
        for r in range(int(citation[i])):
          c=1/(2*publication[i]-1)
        th_list.append(b+c)
    return sum(th_list) 

def Maxprod_index(citation, publication):
    Mi_list=[]
    citation.sort(reverse=True)
    publication.sort()
    for i in range(len(publication)):
      Mi_list.append(citation[i]*publication[i])        
    return max(Mi_list) 

def wu_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  wu_list=[]
  wu1_list=[]
  wu_i=0
  for i in range(len(publication)):
    wu_list.append(10*citation[i])
    wu1_list.append(10*publication[i])
  for j in range(len(publication)):
    if(publication[j]<=wu_list[j]):
      wu_i=j+1
  return wu_i  

def pi_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  pi_index=0
  sq=0
  sum_c=0
  sq=math.sqrt(len(publication))
  for j in range(int(sq)):
    sum_c=sum_c+citation[j]
  pi_index=0.01*sum_c
  return pi_index

def weighted_h_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  hw_index=0
  m=0
  wl_list=[]
  for i in range(len(publication)):
    if(publication[i]<=citation[i]):
      h_index=publication[i]
  for j in range(len(publication)+1):
    sum=0
    for p in range(j):
      sum=sum+citation[p]
    if(h_index==0):
      wl_list.append(0)
    else:
      wl_list.append(sum/h_index)
  wl_list.pop(0)
  for k in range(len(publication)):
    if(wl_list[k]<=citation[k]):
      m=k+1
  for r in range(m):
    hw_index=hw_index+citation[r]
  return math.sqrt(hw_index)

def woginger_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  wo_index=0
  for j in range(len(publication)):
    if(citation[j]>=(len(publication)-publication[j]+1)):
      wo_index=wo_index+1
  return wo_index

def Gh_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  gh_list=[]
  gs_list=[]
  gh_index=0
  h_index=h_index_f(citation,publication)
  for j in range(len(publication)):
    gh_list.append(citation[j]-h_index)
  for k in range(len(publication)):
    if(gh_list[k]>=0):
      gh_index=gh_index+1
  return gh_index

def Rm_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  rm_index=0
  h_index=0
  rm_list=[]
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    rm_list.append(math.sqrt(citation[j]))
  rm_index=math.sqrt(sum(rm_list))
  return rm_index

def x_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  x_list=[]
  ma=0
  sq=0.0
  x_index=0
  for i in range(len(publication)):
      x_list.append(publication[i]*citation[i])
  ma=max(x_list)
  x_index=x_list.index(ma)
  sq=math.sqrt(x_index+1)
  return sq

def h2upper_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h2list=[]
  sum1=0
  sum2=0
  h2upper=0
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    h2list.append(citation[j]-h_index)
  sum1=sum(h2list)
  sum2=sum(citation)
  if(sum2==0):
    h2upper=0
  else:
    h2upper=(sum1/sum2)*100
  return h2upper

def h2center_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  sum1=0
  h2center=0
  h_index=h_index_f(citation,publication)
  sum1=sum(citation)
  if(sum1==0):
    h2center=0
  else:
    h2center=(h_index*h_index/sum1)*100
  return h2center

def h2lower_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h2llist=[]
  sum1=0
  sum2=0
  h2lower=0
  h_index=h_index_f(citation,publication)
  km=h_index+1
  for j in range(km,len(publication)):
    h2llist.append(citation[j]-h_index)
  sum1=sum(h2llist)
  sum2=sum(citation)
  if(sum2==0):
    h2lower=0
  else:
    h2lower=(sum1/sum2)*100
  return h2lower

def k_dash_index(citation, publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  sumh=0
  sumt=0
  k_dash_index=0
  citall=sum(citation)
  pubcount=len(publication)
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    sumh=sumh+citation[j]
  km=h_index+1
  for k in range(km,len(publication)):
    sumt=sumt+citation[k]
  if(sumt==0 or sumh==0 or (sumt-sumh)==0):
    k_dash_index=0
  else:
    k_dash_index=(citall-pubcount)/(sumt-sumh)
  return k_dash_index


def iten_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  count=0
  for i in range(len(publication)):
    if(citation[i]>=10):
      count=count+1
  return count

def normalized_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  nh_index=0
  h_index=0
  pub=0
  pub=len(publication)
  h_index=h_index_f(citation,publication)
  nh_index=h_index/pub  
  return nh_index

def platinium_h_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  ph_index=0
  CL=0
  total_cit=0
  total_pub=0
  CL=max(years)-min(years)
  total_cit=sum(citation)
  total_pub=len(publication)
  h_index=h_index_f(citation,publication)
  if(CL==0):
    ph_index=(h_index)*(total_cit/total_pub)
  else:
    ph_index=(h_index/CL)*(total_cit/total_pub)
  return ph_index

def m_qoutient_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  CL=0
  CL=max(years)-min(years)
  m_quotient=0
  h_index=h_index_f(citation,publication)
  if(CL==0):
    m_quotient=(h_index)
  else:
    m_quotient=(h_index/CL)
  return m_quotient

def HI_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  HI_index=0
  listt=[]
  h_index=h_index_f(citation,publication)
  for j in range(len(publication)):
    listt.append(authors_l[j])
  average_au=sum(listt)/len(listt)
  HI_index=h_index/average_au
  return HI_index

def AW_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  l1=[]
  l2=[]
  summ=0
  papers=1
  for j in range(len(years)):
    summ=0
    papers=0
    for k in range(len(years)):
      if(years[j]==years[k]):
        summ=summ+citation[k]
        papers=papers+1
    l1.append(years[j])
    l2.append(summ/papers)
  return math.sqrt(sum(l2))

def Ar_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  AW_index=0
  h_core_p=[]
  h_core_c=[]
  h_core_y=[]
  h_index=h_index_f(citation,publication)
  for j in range(h_index):
    h_core_p.append(publication[j])
  for c in range(h_index):
    h_core_c.append(citation[c])   
  for y in range(h_index):
    h_core_y.append(years[y])
  cleanedList = [x for x in set(h_core_y) if str(x) != 'nan']
  l1=[]
  l2=[]
  summ=0
  papers=1
  for j in range(len(cleanedList)):
    summ=0
    papers=0
    for k in range(len(h_core_y)):
      if(cleanedList[j]==h_core_y[k]):
        summ=summ+h_core_c[k]
        papers=papers+1
    l1.append(cleanedList[j])
    l2.append(summ/papers)
  return math.sqrt(sum(l2))

def v_index(citation, publication,years):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  import datetime
  today = datetime.date.today()
  cyear = today.year
  CL=0
  CL=cyear-min(years)
  v_index=0
  for i in range(len(publication)):
    if(publication[i]<=citation[i]):
      h_index=publication[i]
  v_index=(h_index/CL)
  return v_index

def hm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  hm_index=0
  weights=[]
  for k in range(len(publication)):
    if (authors_l[k]!=0):
      weights.append(1/authors_l[k])
    else:
      weights.append(1)
  new_list=[]
  j=0
  for i in range(0,len(weights)):
      j+=weights[i]
      new_list.append(j)  
  for m in range(len(publication)):
    if(new_list[m]<=citation[m]):
      hm_index=new_list[m]
  return hm_index

def gm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  eff_rank=0
  ef_rank=[]
  normalized_c=[]
  citation_ss=0
  citation_s=[]
  effe_cit=[]
  gm_index=0
  flag=0
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for m in range(len(publication)):
    effe_cit.append(citation_s[m]/ef_rank[m])
  for l in range(len(publication)):
    if(ef_rank[l]>citation_s[l]):
      gm_index=ef_rank[l-1]
      flag=1
      break
  if(flag==0): 
    gm_index=len(publication)
  return gm_index
  

def hf_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  flag=0
  hf_index=0
  normalized_c=[]
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for k in range(len(publication)):
    if(publication[k]>normalized_c[k]):
      hf_index=k+1
      flag=1
      break
  if(flag==0):
    hf_index=len(publication) 
  return hf_index

def gf_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  gf_index=0
  normalized_c=[]
  flag=0
  rank_s=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for k in range(len(publication)):
    rank_s.append(publication[k]*publication[k])
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(rank_s[l]>=citation_s[l]):
      gf_index=publication[l]
      flag=1
      break
  if(flag==0): 
    gf_index=len(publication)
  return gf_index

def gF_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  gF_index=0
  flag=0
  citation_s=[]
  citation_ss=0
  ef_rank=[]
  eff_rank=0
  eff_rank_square=[]
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for m in range(len(publication)):
    eff_rank_square.append(ef_rank[m]*ef_rank[m])
  for l in range(len(publication)):
    if(eff_rank_square[l]>=citation_s[l]):
      gF_index=publication[l]
      break
  if(flag==0): 
    gF_index=len(publication)
  return gF_index

def hi_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_i_index=0
  coauthor=[]
  coauthors=0
  h_index=h_index_f(citation,publication)
  for k in range(h_index):
    coauthors=coauthors+authors_l[k];
  if(coauthors==0):
    h_i_index=(h_index*h_index);
  else:
    h_i_index=(h_index*h_index)/coauthors;
  return h_i_index

def h_m_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  i=0
  hm_index=0
  coauthor=[]
  flag=0
  less=0
  ef_rank=[]
  eff_rank=0
  for k in range(len(publication)):
    eff_rank=eff_rank+1/authors_l[k]
    ef_rank.append(eff_rank)
  for l in range(len(publication)):
    if(ef_rank[l]>citation[l]):
      hm_index=ef_rank[l]
      break
    elif (ef_rank[l]==citation[l]):
      flag=flag+1
    elif(ef_rank[l]<citation[l]):
      less=less+1
  hm_index=ef_rank[(flag+less)-1]
  return hm_index

def k_norm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  k_norm=0
  h_norm=0
  equal=0
  greater=0
  coauthor=[]
  normalized_c=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(normalized_c[l]<publication[l]):
      h_norm=l
      break
    elif(normalized_c[l]==publication[l]):
      equal=equal+1
    elif(normalized_c[l]>publication[l]):
      greater=greater+1
  h_norm=publication[(greater+equal)-1]
  if(citation_s[h_norm-1]==0):
    k_norm=0
  else:
    k_norm=h_norm+(1-(h_norm*h_norm)/citation_s[h_norm-1])
  return k_norm

def w_norm_index(citation, publication, authors_l):
  citation.sort(reverse=True)
  publication.sort()
  w_norm=0
  h_norm=0
  equal=0
  greater=0
  coauthor=[]
  normalized_c=[]
  citation_s=[]
  citation_ss=0
  for i in range(len(publication)):
    normalized_c.append(citation[i]/authors_l[i])
  normalized_c.sort(reverse=True)
  for n in range(len(publication)):
    citation_ss=citation_ss+citation[n]
    citation_s.append(citation_ss)
  for l in range(len(publication)):
    if(normalized_c[l]<publication[l]):
      h_norm=l
      break
    elif(normalized_c[l]==publication[l]):
      equal=equal+1
    elif(normalized_c[l]>publication[l]):
      greater=greater+1
  h_norm=publication[(greater+equal)-1]
  if(citation_s[len(citation_s)-1]<=0):
    w_norm=0
  else:
    w_norm=h_norm+(1-(h_norm*h_norm)/citation_s[len(citation_s)-1])
  return w_norm

def A_index(citation, publication):
  hcc=0
  h_index=0
  a_index=0
  citation.sort(reverse=True)
  publication.sort()
  hcc=h_core_citation(citation,publication)
  h_index=h_index_f(citation,publication)
  if(h_index==0):
    a_index=0
  else:
    a_index=hcc/h_index
  return a_index

def R_index(citation, publication):
  hcc=0
  R_index=0
  citation.sort(reverse=True)
  publication.sort()
  hcc=h_core_citation(citation,publication)
  R_index=math.sqrt(hcc)
  return R_index

def P_index(citation, publication):
  sc=0
  p_index=0
  citation.sort(reverse=True)
  publication.sort()
  sc=sum(citation)
  p=len(publication)
  p_index=sc/p
  return p_index

def q_sq_index(citation, publication):
  h_index=0
  m_index1=0
  q_sq_index1=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  m_index1=m_index(citation,publication)
  q_sq_index1=math.sqrt(h_index*m_index1)
  return q_sq_index1

def e_index(citation, publication):
  h_index=0
  e_index=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  hcc=h_core_citation(citation,publication)
  e_index=hcc-(h_index*h_index)
  return e_index

def hg_index(citation, publication):
  h_index=0
  g_index=0
  hg_index=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  g_index=g_index_f(citation,pub_id)
  hg_index=math.sqrt(h_index*g_index)
  return hg_index

def pure_h_index(citation, publication,author):
  h_index=0
  author_c=0
  author_av=0
  pure_h_indexx=0
  citation.sort(reverse=True)
  publication.sort()
  h_index=h_index_f(citation,publication)
  for i in range(h_index):
    author_c=author_c+author[i]
  if(h_index==0):
    author_av=0
  else:
    author_av=author_c/h_index
  if(author_av==0):
    pure_h_indexx=0
  else:
    pure_h_indexx=h_index/math.sqrt(author_av)
  return pure_h_indexx

def fractional_g_index_f(citation,publication,author):
  frac_g_index=0
  citation.sort(reverse=True)
  publication.sort()
  pub_s=[]
  c_su=0
  c_sum=[]
  pub_s=[i ** 2 for i in publication]
  for i in range(len(publication)):
    c_su=c_su+citation[i]/author[i]
    c_sum.append(c_su)  
  for i in range(len(publication)):
    if (len(publication)==1 and citation[i]>=1):
      frac_g_index=1
    elif (len(publication)==1 and citation[i]==0):
      frac_g_index=0
    elif(pub_s[i]<=c_sum[i]):
      frac_g_index=publication[i]
  return frac_g_index

def fractional_h_index_f(citation,publication,author):
  frac_h_index=0
  citation.sort(reverse=True)
  publication.sort()
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=citation[i]/author[i]
    c_ss.append(c_s) 
  for j in range(len(publication)):
    if (len(publication)==1 and c_ss[j]>=1):
      frac_h_index=1
    elif (len(publication)==1 and c_ss[j]==0):
      frac_h_index=0
    elif(publication[j]<=c_ss[j]):
      frac_h_index=publication[j]
  return frac_h_index

def AWCR_index_f(citation,publication,year):
  awcr_index=0
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_d=0
  c_s=0
  sum=0
  for i in range(len(publication)):
    c_d=cyear-year[i]
    if(c_d!=0):
      c_s=citation[i]/c_d
    else:
      c_s=citation[i]/1
    awcr_index=awcr_index+c_s
  return awcr_index

def Author_paper(publication,authors):
  author_p=0
  author=0
  paper=0
  author=sum(authors)
  paper=len(publication)
  author_p=author/paper
  return author_p

def cites_author(citation, publication,authors):
  citation.sort(reverse=True)
  publication.sort()
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=citation[i]/authors[i]
    c_ss.append(c_s) 
  return sum(c_ss)

def papers_author(publication,authors):
  publication.sort()
  c_s=0
  c_ss=[]
  for i in range(len(publication)):
    c_s=1/authors[i]
    c_ss.append(c_s) 
  return sum(c_ss)

def k_index(citation,publication,authors):
  citation.sort(reverse=True)
  publication.sort()
  k_index=0
  c_t=0
  p_t=0
  h_index=0
  c_total=sum(citation)
  p_total=len(publication)
  h_index=h_index_f(citation,publication)
  c_s=0
  c_core=0
  c_t=0
  c_tail=0
  for i in range(h_index):
    c_s=c_s+citation[i]
  c_core=c_s
  for i in range(h_index, len(publication)):
    c_t=c_t+citation[i]
  c_tail=c_t
  if(c_tail==0):
    if(c_core==0):
      k_index=0
    else:
      k_index=(c_total/p_total)/c_core
  else:
    k_index=(c_total/p_total)/(c_tail/c_core)
  return k_index

def h_dash_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_dash_indexx=0
  e_indexx=0
  t_indexx=0
  h_indexx=0
  e_indexx=e_index(citation, publication)
  t_indexx=t_index(citation, publication)
  h_indexx=h_index_f(citation,publication)
  h_dash_indexx=(e_indexx/t_indexx)*h_indexx
  return h_dash_indexx

def hl_norm(citation,publication,author):
  citation.sort(reverse=True)
  publication.sort()
  c_t=[]
  h_norm=0
  for i in range(len(publication)):
    c_t.append(citation[i]/author[i])
  c_t.sort(reverse=True)
  h_norm=h_index_f(c_t,publication)
  return h_norm

def hc_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_t=[]
  sc=0
  hc_index=0
  for i in range(len(publication)):
    sc=4*(citation[i]/(cyear-years[i]+1))
    c_t.append(sc)
  c_t.sort(reverse=True)
  hc_index=h_index_f(c_t,publication)
  return hc_index

def ha_index(citation,publication,years):
  citation.sort(reverse=True)
  publication.sort()
  import datetime
  today = datetime.date.today()
  cyear = today.year
  c_t=[]
  sc=0
  ha_indexx=0
  for i in range(len(publication)):
    if(cyear-years[i]!=0):
      sc=citation[i]/(cyear-years[i])
    else:
      sc=citation[i]
    c_t.append(sc)
  c_t.sort(reverse=True)
  ha_indexx=h_index_f(c_t,publication)
  return ha_indexx

def hi_index_s(citation,publication,authors):
  citation.sort(reverse=True)
  publication.sort()
  h_indexx=0
  h_indexx=h_index_f(citation,publication)
  hi_index_ss=0
  c_t=0
  c=0
  sc=0
  ha_indexx=0
  for i in range(h_indexx):
    c=c+1
    sc=sc+authors[i]
  if(c==0):
    c_t=0
    hi_index_ss=0
  else:
    c_t=sc/c
    hi_index_ss=h_indexx/c_t
  return hi_index_ss

def rational_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_index=h_index_f(citation,publication)
  k=0
  h_r_index=0
  if(h_index==len(publication)):
    k=citation[h_index-1]
    h_r_index=(h_index+1)-k/(2*h_index+1)
  else:
    k=citation[h_index]-citation[h_index-1]
    h_r_index=(h_index+1)-k/(2*h_index+1)
  return h_r_index

def real_h_index(citation,publication):
  citation.sort(reverse=True)
  publication.sort()
  h_index=0
  h_index=h_index_f(citation,publication)
  k=0
  z=0
  r_h_index=0
  if(h_index==len(publication)):
    k=(h_index+1)* citation[h_index-1] - h_index*citation[h_index-1]+1 
    z=1-citation[h_index-1]
  else:
    k=(h_index+1)* citation[h_index-1] - h_index*citation[h_index-1]+1 
    z=1-citation[h_index] + citation[h_index-1]
  if(z==0):
    r_h_index= 0
  else:
    r_h_index= k/z
  return r_h_index


def Author_indexs_c(pub_id,citation,author_name,publication_y,co_auth):
  row=[]
  row.append(author_name)
  row.append(publication_count(pub_id))
  row.append(citation_count(citation))
  row.append(total_years(publication_y))
  row.append(cite_per_years(citation,publication_y))
  row.append(cite_per_paper(citation,pub_id))
  row.append(h_index_f(citation,pub_id))
  row.append(g_index_f(citation,pub_id))
  row.append(h2_index_f(citation,pub_id))
  row.append(w_index_f(citation,pub_id))
  row.append(h_core_citation(citation,pub_id))
  row.append(m_index(citation,pub_id))
  row.append(f_index(citation,pub_id))
  row.append(t_index(citation,pub_id))
  row.append(tappered_h_index(citation,pub_id))
  row.append(Maxprod_index(citation,pub_id))
  row.append(wu_index(citation,pub_id))
  row.append(pi_index(citation,pub_id))
  row.append(weighted_h_index(citation,pub_id))
  row.append(woginger_index(citation,pub_id))
  row.append(Gh_index(citation,pub_id))
  row.append(Rm_index(citation,pub_id))
  row.append(x_index(citation,pub_id))
  row.append(h2upper_index(citation,pub_id))
  row.append(h2center_index(citation,pub_id))
  row.append(h2lower_index(citation,pub_id))
  row.append(k_dash_index(citation,pub_id))
  row.append(iten_index(citation,pub_id))
  row.append(normalized_h_index(citation,pub_id))
  row.append(platinium_h_index(citation,pub_id,publication_y))
  row.append(m_qoutient_index(citation,pub_id,publication_y))
  row.append(HI_index(citation,pub_id,co_auth))
  row.append(AW_index(citation,pub_id,publication_y))
  row.append(Ar_index(citation,pub_id,publication_y))
  row.append(v_index(citation,pub_id,publication_y))
  row.append(hm_index(citation,pub_id,co_auth))
  row.append(hf_index(citation,pub_id,co_auth))
  row.append(gf_index(citation,pub_id,co_auth))
  row.append(gF_index(citation,pub_id,co_auth))
  row.append(hi_index(citation,pub_id,co_auth))
  row.append(h_m_index(citation,pub_id,co_auth))
  row.append(k_norm_index(citation,pub_id,co_auth))
  row.append(w_norm_index(citation,pub_id,co_auth))
  row.append(gm_index(citation,pub_id,co_auth))
  row.append(A_index(citation,pub_id))
  row.append(R_index(citation,pub_id))
  row.append(P_index(citation,pub_id))
  row.append(q_sq_index(citation,pub_id))
  row.append(e_index(citation,pub_id))
  row.append(hg_index(citation,pub_id))
  row.append(pure_h_index(citation,pub_id,co_auth))
  row.append(fractional_g_index_f(citation,pub_id,co_auth))
  row.append(fractional_h_index_f(citation,pub_id,co_auth))
  row.append(AWCR_index_f(citation,pub_id,publication_y))
  row.append(Author_paper(pub_id,co_auth))
  row.append(cites_author(citation, pub_id,co_auth))
  row.append(papers_author(pub_id,co_auth))
  row.append(k_index(citation, pub_id,co_auth))
  row.append(h_dash_index(citation, pub_id))
  row.append(hl_norm(citation,pub_id,co_auth))
  row.append(hc_index(citation,pub_id,publication_y))
  row.append(ha_index(citation,pub_id,publication_y))
  row.append(hi_index_s(citation,pub_id,co_auth))
  row.append(rational_h_index(citation,pub_id))
  row.append(real_h_index(citation,pub_id))
  print(row)
  list_l.append(row)
  


data = pd.read_csv('UpdateAwardeesmerge.csv',encoding='latin-1')
authors_name=[]
publication=[]
citation=[]
publishing_years=[]
co_author=[]
for col in data.itertuples(): #reading file
  authors_name.append(col[1])
  publication.append(col[2])
  citation.append(col[3])
  publishing_years.append(col[4])
  co_author.append(col[5])
author_name=''
pub_id=[]
citation_c=[]
publication_y=[]
co_auth=[]
dff = pd.DataFrame(authors_name, columns=['author_name'])
gk=dff.groupby('author_name').agg(lambda x: ','.join(x))
author_l=gk.index.tolist()
k=0
for j in range(len(author_l)):
  for r in range(len(authors_name)):
    author_name=author_l[j]
    if(author_l[j]==authors_name[r]):
      pub_id.append(publication[r])
      citation_c.append(citation[r])
      publication_y.append(publishing_years[r])
      co_auth.append(co_author[r])
    else:
      continue
  k=k+1
  print(k , author_name)
  citation_c=[int(a) for a in citation_c]
  publication_y=[int(a) for a in publication_y]
  co_auth=[int(a) for a in co_auth]
  print(citation_c)
  Author_indexs_c(pub_id,citation_c,author_name,publication_y,co_auth)
  pub_id=[]
  citation_c=[]
  publication_y=[]
  co_auth=[]

file = open('UpdateAwardeesmerge_file.csv', 'w+', newline ='')
header = [['Author Name', 'Total Publication', 'Total Citation','Total Years','Cite/Year','Cite/Paper','h index','g index',
           'h(2) index', 'w index','h core citation', 'm index', 'f index', 't index', 'tappered_h_index', 'Maxprod',
           'wu index', 'pi index', 'weighted h index','woginger_index', 'Gh_index', 'Rm index', 'X index','h2upper_index',
           'h2 center_index','h2 lower_index','k dash index','i10 index','normalized_h_index','platinium_h_index',
           'm_qoutient_index','HI_index','AW_index','Ar index','V index','hm_index','hf index','gf index','gF index',
           'hi index','h_m_index','k norm index','w norm index','gm index','A index','R index','P index','q2 index',
           'e index', 'hg_index','pure h index','fractional g index','fractional h index','AWCR','Author/paper',
           'cites/author','paper/author','k index','h dash index','hl norm','hc index','ha index','hi index s',
           'Rational h index', 'Real h index']]
with file:   
  write = csv.writer(file)
  write.writerows(header)
  write.writerows(list_l)


"""Configurable character chunking and heading/paragraph/sentence-aware chunking."""
import re
from dataclasses import dataclass
from typing import Literal

Strategy = Literal['fixed', 'structure']

@dataclass
class Chunk:
    text: str
    start: int
    end: int
    boundary: str

def validate(chunk_size: int, overlap: int, strategy: str):
    if strategy not in ('fixed', 'structure'):
        raise ValueError('strategy must be fixed or structure')
    if not 50 <= chunk_size <= 10000:
        raise ValueError('chunk_size must be between 50 and 10000 characters')
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError('overlap must be >= 0 and smaller than chunk_size')

def _heading(line: str) -> bool:
    s=line.strip()
    return bool(re.match(r'^#{1,6}\s+|^(?:\d+(?:\.\d+)*[.)]?\s+)[A-Z]|^(?:chapter|section|appendix)\s+\w+',s,re.I)) or (4<=len(s)<=85 and s.isupper() and any(c.isalpha() for c in s))

def _boundaries(text: str):
    """Candidate cuts (end offset, priority, label), always inside original text."""
    found=[]
    for m in re.finditer(r'\n[^\S\n]*\n+',text):found.append((m.end(),3,'paragraph'))
    for m in re.finditer(r'[^\n]+(?:\n|$)',text):
        if _heading(m.group().strip()):
            found.append((m.start(),4,'heading'))
            found.append((m.end(),4,'heading'))
    for m in re.finditer(r'[.!?](?:["\'”’)]*)\s+',text):found.append((m.end(),2,'sentence'))
    for m in re.finditer(r'\s+',text):found.append((m.end(),1,'word'))
    return sorted(set((p,priority,label) for p,priority,label in found if 0<p<len(text)))

def chunk_text(text: str, strategy: Strategy='fixed', chunk_size: int=950, overlap: int=150) -> list[Chunk]:
    validate(chunk_size,overlap,strategy)
    if not isinstance(text,str):raise TypeError('text must be a string')
    if not text.strip():return []
    candidates=_boundaries(text) if strategy=='structure' else []
    result=[]; start=0; length=len(text)
    while start<length:
        target=min(start+chunk_size,length)
        end=target; boundary='fixed' if strategy=='fixed' else 'size limit'
        if strategy=='structure' and target<length:
            options=[c for c in candidates if start+max(1,int(chunk_size*.45))<=c[0]<=target]
            if options:
                # Choose highest-priority boundary, then the latest such boundary.
                p,_,boundary=max(options,key=lambda c:(c[1],c[0]));end=p
        a=start;b=end
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:result.append(Chunk(text[a:b],a,b,boundary))
        if end>=length:break
        next_start=max(start+1,end-overlap)
        while next_start<length and text[next_start].isspace():next_start+=1
        start=next_start
    return result

def report(text: str, strategy: Strategy='fixed', chunk_size: int=950, overlap: int=150):
    chunks=chunk_text(text,strategy,chunk_size,overlap)
    return {'strategy':strategy,'chunk_size':chunk_size,'overlap':overlap,
            'chunk_count':len(chunks),
            'average_chars':round(sum(len(c.text) for c in chunks)/len(chunks),1) if chunks else 0,
            'chunks':[{'index':i+1,'start':c.start,'end':c.end,'characters':len(c.text),
                       'boundary':c.boundary,'text':c.text,
                       'shared_with_previous':max(0,min(c.end,chunks[i-1].end)-max(c.start,chunks[i-1].start)) if i else 0}
                      for i,c in enumerate(chunks)]}

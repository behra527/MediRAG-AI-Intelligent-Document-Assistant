from app.rag import split_text

def test_empty(): assert split_text('   ')==[]
def test_overlap():
    chunks=split_text(' '.join(['diabetes']*300),100,20)
    assert len(chunks)>1
    assert all(len(c)<=100 for c in chunks)

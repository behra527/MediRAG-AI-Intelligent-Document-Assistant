from app.services.chunking import chunk_text, report
import pytest

def test_fixed_offsets_and_overlap():
    text='abcdefghijklmnopqrstuvwxyz'*8
    chunks=chunk_text(text,'fixed',50,10)
    assert chunks[0].text==text[:50]
    assert chunks[1].start==40
    assert chunks[1].text==text[40:90]
    assert chunks[-1].end==len(text)

def test_structure_prefers_paragraph():
    text='First paragraph. ' * 5+'\n\n'+'Second paragraph. '*5
    result=chunk_text(text,'structure',105,0)
    assert result[0].boundary in ('paragraph','sentence')
    assert all(text[c.start:c.end]==c.text for c in result)

def test_invalid_overlap():
    with pytest.raises(ValueError):chunk_text('text','fixed',50,50)

def test_experiment_report():
    data=report('a'*180,'fixed',80,20)
    assert data['chunk_count']==3
    assert data['chunks'][1]['shared_with_previous']==20

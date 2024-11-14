from typing import List, Tuple, Dict
testset: Dict[str, Tuple] = {
     "cat" : ("catscsctadtcxcacsttacaactax", 5)
}

def solution(anagram: str, words: str):
     ...
    
for test in testset:
    assert testset[test][1] == solution(anagram=test, words=testset[test][0])
    print("정답입니다!")
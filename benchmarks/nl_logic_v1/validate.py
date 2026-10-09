"""Validate integrity and open-split ground truth without opening heldout."""
import argparse
import collections
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from logic import solve, closure, to_problog, prolog_atom
from score import read_jsonl, evaluate

ROOT = Path(__file__).resolve().parent


def self_checks():
    def choice(name,p):
        return {"id":name,"outcomes":[{"p":str(p),"facts":[[name]]},{"p":str(1-p),"facts":[]}]}
    def rule(head,*body):
        return {"head":[head],"body":[[x] for x in body]}
    base = {"facts":[],"choices":[choice("a",Fraction(1,5)),choice("b",Fraction(3,10)),choice("c",Fraction(1,2))],"queries":[{"id":"q","atom":["q"]}],"evidence":[]}
    fixtures = [
        ([rule("q","a","a")],[],Fraction(1,5)),
        ([rule("q","a","b")],[],Fraction(3,50)),
        ([rule("q","a"),rule("q","b")],[],Fraction(11,25)),
        ([rule("q","a","b"),rule("q","a","c")],[],Fraction(13,100)),
        ([rule("alarm","a"),rule("alarm","b"),rule("q","a")],[{"atom":["alarm"],"value":True}],Fraction(5,11)),
        ([rule("q","q"),rule("q","a")],[],Fraction(1,5)),
    ]
    for rules,evidence,expected in fixtures:
        actual = solve({**base,"rules":rules,"evidence":evidence},[])
        assert Fraction(actual["query_probabilities"]["q"]) == expected,(actual,expected)
    facts = [["edge","a","b"],["edge","b","a"],["p","a"],["not_p","a"]]
    rules = [
        {"head":["reach","?x","?y"],"body":[["reach","?x","?z"],["edge","?z","?y"]]},
        {"head":["reach","?x","?y"],"body":[["edge","?x","?y"]]},
    ]
    result = closure(facts,rules)
    assert ("reach","a","a") in result and ("reach","b","b") in result
    assert ("unrelated","a") not in result
    exclusive = {"facts":[],"choices":[{"id":"mode","outcomes":[{"p":"1/3","facts":[["x"]]},{"p":"2/3","facts":[["y"]]}]}],"rules":[rule("q","x","y")],"evidence":[],"queries":[{"id":"q","atom":["q"]}]}
    assert solve(exclusive,[])["query_probabilities"]["q"] == "0"
    return len(fixtures)+2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problog",action="store_true",help="Cross-check each open underlying world against ProbLog")
    parser.add_argument("--self-checks-only",action="store_true")
    args=parser.parse_args()
    analytical_count=self_checks()
    if args.self_checks_only:
        print(json.dumps({"analytical_fixtures":analytical_count,"passed":True}));return
    manifest=json.loads((ROOT/"manifest.json").read_text())
    unavailable_local_artifacts=[]
    for relative,expected in manifest["sha256"].items():
        artifact=ROOT/relative
        if not artifact.exists() and relative.startswith("sealed/"):
            unavailable_local_artifacts.append(relative)
            continue
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==expected,relative
    all_ids=set(); group_splits={}; checks=0; problog_checks=0; max_error=0.; status_counts=collections.Counter(); query_counts=collections.Counter()
    for split in ("train","dev"):
        inputs=read_jsonl(ROOT/"data"/f"{split}.inputs.jsonl")
        gold=read_jsonl(ROOT/"data"/f"{split}.gold.jsonl")
        assert set(inputs)==set(gold)
        assert not (all_ids & set(inputs)); all_ids.update(inputs)
        groups=collections.defaultdict(list)
        for pid,inp in inputs.items():
            assert inp["group_id"] not in group_splits or group_splits[inp["group_id"]]==split
            group_splits[inp["group_id"]]=split
            groups[inp["group_id"]].append(pid)
            target=gold[pid]
            actual=solve(inp["program"],target["oracle_facts"])
            assert actual["query_probabilities"]==target["query_probabilities"]
            assert actual["evidence_probability"]==target["evidence_probability"]
            facts=set(map(tuple,target["oracle_facts"]))
            for candidate in inp["candidates"]:
                pos=tuple(candidate["atom"]) in facts; neg=tuple(candidate["negative_atom"]) in facts
                state="both" if pos and neg else "supported" if pos else "refuted" if neg else "unknown"
                assert state==target["assessments"][candidate["id"]]
                status_counts[state]+=1
            for p in actual["query_probabilities"].values():
                v=Fraction(p); assert 0<=v<=1
                query_counts["zero" if v==0 else "one" if v==1 else "fractional"]+=1
            checks+=1
        for ids in groups.values():
            assert len(ids)==2
            a,b=ids
            assert inputs[a]["program"]==inputs[b]["program"]
            assert inputs[a]["world_text"]!=inputs[b]["world_text"]
            for field in ("assessments","query_probabilities","oracle_facts"):
                assert gold[a][field]==gold[b][field]
            if args.problog:
                from problog import get_evaluatable
                from problog.program import PrologString
                inp,target=inputs[a],gold[a]
                source=to_problog(inp["program"],target["oracle_facts"])
                result=get_evaluatable().create_from(PrologString(source)).evaluate()
                # Parse the exported query terms so quotes render exactly as ProbLog does.
                queries=list(PrologString("\n".join(prolog_atom(q["atom"])+"." for q in inp["program"]["queries"])))
                for q,term in zip(inp["program"]["queries"],queries):
                    error=abs(result[term]-float(Fraction(target["query_probabilities"][q["id"]])))
                    assert error<1e-9,(inp["family"],q,error)
                    max_error=max(max_error,error)
                problog_checks+=1
        sanity=evaluate(inputs,gold,gold)
        assert sanity["overall"]["assessment_accuracy"]==1
        assert sanity["overall"]["query_probability_mae"]==0
        derived=evaluate(inputs,gold,gold,derive_probabilities=True)
        assert derived["overall"]["query_probability_mae"]==0
    report={
        "analytical_fixtures":analytical_count,"open_programs_checked":checks,
        "problog_worlds_checked":problog_checks,"problog_max_absolute_error":max_error,
        "open_assessment_counts":dict(status_counts),"open_query_counts":dict(query_counts),
        "all_available_manifest_checksums_valid":True,
        "unavailable_local_artifacts":unavailable_local_artifacts,
        "heldout_archive_opened":False,
        "heldout_model_evaluations":0,
    }
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()

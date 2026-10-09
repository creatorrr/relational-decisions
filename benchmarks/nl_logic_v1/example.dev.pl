'entity'('maple_02_6_0').
'entity'('maple_02_6_1').
'entity'('maple_02_6_2').
'entity'('maple_02_6_3').
'needs_attention'('maple_02_6_0').
'ready'('maple_02_6_0').
'requested'('maple_02_6_0').
'needs_attention'('maple_02_6_1').
'requested'('maple_02_6_1').
'not_requested'('maple_02_6_1').
'not_needs_attention'('maple_02_6_2').
'ready'('maple_02_6_2').
'not_needs_attention'('maple_02_6_3').
'ready'('maple_02_6_3').
'not_ready'('maple_02_6_3').
'not_requested'('maple_02_6_3').
'gate'('g0') :- 'choice_g0'('0').
0.20000000000000001::'choice_g0'('0'); 0.80000000000000004::'choice_g0'('1').
'gate'('g1') :- 'choice_g1'('0').
0.69999999999999996::'choice_g1'('0'); 0.29999999999999999::'choice_g1'('1').
'gate'('g2') :- 'choice_g2'('0').
0.80000000000000004::'choice_g2'('0'); 0.20000000000000001::'choice_g2'('1').
'left'(V_x) :- 'needs_attention'(V_x), 'gate'('g0'), 'gate'('g1').
'right'(V_x) :- 'requested'(V_x), 'gate'('g0'), 'gate'('g2').
'act'(V_x) :- 'left'(V_x).
'act'(V_x) :- 'right'(V_x).
query('left'('maple_02_6_0')).
query('right'('maple_02_6_0')).
query('act'('maple_02_6_0')).
query('act'('maple_02_6_1')).
query('act'('maple_02_6_2')).
query('act'('maple_02_6_3')).

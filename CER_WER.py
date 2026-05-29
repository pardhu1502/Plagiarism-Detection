from jiwer import wer, cer
ground_truth = """
Submitted by:
Roll: 22BCS002
Name: Abhijay
V+th Sem, CSE'A'
CS304 x Artificial Intelligence
Assignment - 2

1 Ans We can use forward checking to prove the required statement.
Step I: Conversion of given facts into FOL:
① American (a) ∧ weapon (b) ∧ sells (a, b, c) ∧ hostile (c)
→ Criminal (a)
(Where a, b, c, d are variables)
② Owns (A, m)
Using Existential Instantiation
to write Country A has some Missiles
③ Missile (m)
④ ∀: Missile (m) ∧ Owns (A, m) → Sells (Robert, m, x)
⑤ Missile (m) → Weapons (m)
⑥ Enemy (Z, America) → Hostile (z)
⑦ Enemy (A, America)
⑧ American (Robert)
Applying Forward Checking
Step I:
American (Robert)
Missile (m)
Owns (A, m)
Enemy (A, America)


===== PAGE 2 =====
Step 2:-
Weapons(m)
Sells(Robert,m,A)
Hostile(A)
American(Robert)
Missile(m)
Owns(A,m)
Enemy(A,America)

Step 3:-
Rimind(Robert)
Weapon(m)
Sells(Robert,m,A)
Hostile(A)
American(Robert)
Missile(m)
Owns(A,m)
Enemy(A,America)

2Ans (Converting given facts to FOL:-
1. ∀x : food(x) → likes(John,x)
2. food(Apple) ∧ food(vegetables)
3. ∀x ∀y : eats(x,y) ∧ killed(x) → food(y)
4. eats(Anil, Peanuts) ∧ alive(Anil)
5. ∀x : eats(Anil,x) → eats(Harry,x)

Added predicates:-
∀x : ¬killed(x) → alive(x)
∀x : alive(x) → ¬killed(x)

Statement to prove:-
likes(John, peanuts)


===== PAGE 3 =====
Step 2: - Converting FOL into (NF

① ¬food(x) V likes(John, x)

② food(Apple)

③ food(Vegetables)

④ ¬eats(y, z) V killed(y) V food(z)

⑤ eats(Anil, Peanuts)

⑥ alive(Anil)

⑦ ¬eats(Anil, w) V eats(Harry, w)

⑧ killed(g) V alive(g)

⑨ ¬alive(k) V → killed(k)

To prove:¬ likes(John, Peanuts)

Step 3: We can prove the statement "likes(John, Peanuts)" by using resolution to prove the statement ("¬likes(John, Peanuts)") is unsatisfiable:

Step 4: Draw resolution graph

¬likes(John, Peanuts) ¬food(x) V likes(John, x)
¬food(Peanuts) ¬eats(y, z) V killed(y) V food(z)
¬eats(y, Peanuts) V killed(y) eats(Anil, Peanuts)
killed(Anil) ¬alive(k) V → killed(k)
¬alive(Anil) alive(Anil)

Hence Proved


===== PAGE 4 =====
3 Ans]
Step I: - Converting given facts into FOL:
① ∀c : easy(c) → likes(Amit, c)
② ∀c : since(c) → Hard(c)
③ ∀c : BW(c) → easy(c)
④ BW(Bk-304)

Added predicates:
① ∀c : Hard(c) → ¬easy(c)
② ∀c : easy(c) → ¬Hard(c)

Step 2: - Converting into CNF:
① ¬easy(c) ∨ likes(Amit, c)
② ¬science(c) ∨ Hard(c)
③ ¬BW(c) ∨ easy(c)
④ BW(Bk-304)
⑤ ¬Hard(c) ∨ easy(c)

Step 3: - Now we prove that "Amit likes courses in Basket weaving department" which we prove by using resolution to prove "¬likes(Amit, c) ∧ BW(c)" is unsatisfiable:

Step 4: - Draw resolution graph:
¬likes(Amit, c) ∧ BW(c) ⊢ easy(c) ∨ likes(Amit, c)
¬likes(Amit, c) ∧ BW(c) ⊢ easy(c) ∨ ¬easy(c)
¬BW(c) ∨ ¬easy(c) ⊢ BW(c) ∨ easy(c)

∴ Hence proved "likes(Amit, c) ∧ BW(c)"
i.e., Amit likes courses in Basket weaving department.
Now BW(Bk-304), Amit likes(Amit, c) ∧ BW(c)
∴ Amit likes(Bk-304) course.
"""

extracted_text = """

===== PAGE 1 =====
Submitted by
Rodl: 22BCS002
CS304 Axtificial Trdelligence
Nome l Abhijay
X
V+hSem CSEA
Assighment I 2
1Ans We (an Use Boswand checking to prove the povired
stdement.
StepT:- Convession d given bads into FOL:-
D Americon Ca) A Weapon (b) 1 sells Ca,b c)A hostilelc)
> Cximinal(a)
(when a, ) bit,d all. variables)
Inslartiation
2 OWnS (A,m)
Opites Using Existmial A has Some Missiles )
- tow wolte (ountry
3 Missile (m)
@ Yo:Missilelm) A Owns (A,0) -> Sells (Rolrt,o,)
5
Missile (m) - >> Weapons (m)
6
Enemyl2, America) - > Hostilelz)
Enemy (A, Amexica)
8
Americon (Robext)
Applyjing Fooward Chrcking
SHtp :
American(Robert) Missilelm)
TOwAm Enomls Amoial

===== PAGE 2 =====
Step2:
V Weapohs (m)
Sello(RoletmyA)
HostilelA)/
Amesican(Robet) Missie(m) Owns(Am)
Enery(A,Antyica))
SHep3:
Hine psovs
Cominel(Robert)
AstileCA)
SeloRomimA
Wepontonll
Americon/Roet) Missilelon) Owrs(Am)
EvoglAAperial
to FOL:- (SHepT)
2Ans
Converting given focts
X )
1. Yx: foallx) > likes (John,
2. foodCApple) A food (vegetobles)
> food - y)
3. Fx Yy: eats (yy) A-killlw
Pats ( Anid, Pranuts) A alive CAnil)
4.
radsCAnd, x ) eats I Harry, 1) -
5. Vr:
Added pordicates:-
(w) > alive(n)
fx: - 1 killed
I (x)
Yx: alielo-skilld
Stodrment to pYove :
likest John, peanuts

===== PAGE 3 =====
Step2 : a Convesting FOL into CNF
@ foodlx) V likes ( John,)
food CApple)
3)
food ( vegetdbles)
4
Teats (y 2) V killedly) V tood(z)
5
eats ( Anil, Peanats)
6
alive (( Anil)
7
7 eats(Anilw) Veats(Hay,)
8
k Wdly) V alivelg)
0
74 alivelk) V-killed(k)
Topsove : /
likes (John, Peanuts) I
the stadement NC, likes (John Peanuts) by using
Hep3:Wt (an poove
Ysolution to poove the Stadement ( likes(John, Peanuts) I 1 is
unsatisfiable:-
Stept:- Dyow aesolution ghaph
3 Dikes (John, Peanuts)
nfoodls) V likes(John >
pranutsln
food ( I Peanuts)
eats (y 2) V killedly) V food (2)
I
Ipeanuslzy
7 fats (y, erwad)vkillalg)
lats ( Anil, Peanuts)
y Anillys
alive (k) V- killedlk)
Kille(Anil)
SAndIk)
70 aliv(Anil)
alivelAnil)
I
- 3
Honte Pxoved

===== PAGE 4 =====
3 Ans I StepI: L Convesting givm facts into FOL :
@ fc: pusylo > likesl Amit, C
2 +c: Smce (c) > Hardlc)
3 fc:
BW(c) > easg (C)
@
BWC8K-304)
Added phedicates : -
fc: Handlo
> 7 pasy (c)
FC:
Pasylo) ) 1 Hasdlc)
Step2:- Corverting into
CNF: -
D
1 pasy (c) a V likes (Amit, c)
2
sciencelc) V Hard(c)
3
BWIC) V pasglc)
4
BW(BK-304)
S
1 Hardle) Krasy (c)
Basket
that Amit likes cowses in
Step3:
Now WP psove
pesolution
wpawing department" which WP prove hy wing
to phove (I - likes (Amit, C) AVBWCC 17 i8 LA
unsatichiable:
Sfepg: : - - Dyaw resolution-gpaph.
7 pasylc) vlikes (Amit, c)
likes (Amit, DavBuc)
BWIC) VI pasy (C)
BW (c) V pasy (C)
13
L( likes (Amit, OABWLO 1)
Hence proved
ire, Amit likes (OWBeS in Baskef Wraving
department
Now
BWIBK-3 04)
AprocA Rikes(Anit,C) ABWIC
/
ARD likes(Amit, Bk-309)
Hence Amit likes Bk-304 coursp.


"""
cer_score = cer(ground_truth, extracted_text)

wer_score = wer(ground_truth, extracted_text)

print(f"CER: {cer_score:.4f}")
print(f"WER: {wer_score:.4f}")

print(f"\nCER Percentage: {cer_score * 100:.2f}%")
print(f"WER Percentage: {wer_score * 100:.2f}%")
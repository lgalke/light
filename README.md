# Light

## Problem Definition

Im vorliegenden Fall bildet das Modell den Zusammenhang zwischen
einer gemessenen Zeit X (in Sekunden) und der Beleuchtungsstärke 
Y (in Lux) an einem Photowiderstand R (in Ohm) ab.

Zu finden sind die unbekannten Modellparameter R, a, b bzw. weitere
Modellparameter, die aus einer Verfeinerung des Modells resultieren.

```
     Y = a * R^-b
```

Eine Datei mit Paaren für X , Y steht dazu als Datei zur Verfügung.

Das Modell beschreibt zunächst nur den Zusammenhang zwischen dem
Widerstand R und der Beleuchtungsstärke L. 

Der Widerstand R kann allerdings nicht gemessen werden. 
Stattdessen wird die Zeit X gemessen, in der ein Kondensator über den 
Photowiderstand aufgeladen wird, bis 
am Eingangspin eines messenden Rechners eine logische "1" angliegt.
In der Regel ist das schon der Fall, bevor der Kondensator C voll 
aufgeladen ist. Aus dem Zusammenhang  R·C = t lässt sich der Widerstand R 
der Photozelle berechnen: R= t/C. C ist der Anteil der Kapazität des 
Kondensators, bei dem der messende Rechner eine logische "1" erkennt.

```

              -
              |
Kondensator  ===
              |
              ·-----> GPIO -->
              |
             ---
Phtozelle    | |
             ---
              |
              +
```

Das Modell hat damit die folgende Form:

```
     Y = a * R^-b, mit R= X/C    als Javascript: y= a*Math.pow(X/C,-b) 
```

Die Modellparameter a,b und C sind unbekannt / nicht genau bekannt.
Ein Näherungswert für C ist die Nennkapazität des 
Kondensators: 100 nF = 1e-7 F, a und b sind positiv.

Eine Verfeinerung des Modells ist dem Umstand geschuldet, dass
der Widerstand der Photozelle nur Werte zwischen einem kleinsten 
Wert r (dem Hellwiderstand) und einem größten Wert r+d (dem Dunkelwiderstand)
annehmen kann. Das Modell nimmt dann die folgende Form an:

```
     Y = a * R^-b, mit R= 1/(1/(X/C-r)-1/d)
```

Auch die weiteren Modellparameter r und d sind unbekannt, typische Werte
sind 100 Ohm für r und 4000000-r Ohm für d.

Die Datei mit Messwerten (tab getrennte csv Datei) enthält in der ersten 
Spalte die gemessene Zeit t in Mikrosekunden, 
die mit X= t/1000000 in Sekunden umzurechnen sind,
die zweite Spalte die gemessene Beleuchtungsstärke Y in Lux.

Zu finden sind die Modellparameter a, b, C, r und d.


## Model outputs 

Parameters `(a, b, C)` estimated using SGD with Nesterov momentum

![Parameters (a,b, C) estimated using SGD with Nesterov momentum][out/math-3.png]

100 hidden units MLP optimized by LBFGS

![100 hidden units MLP optimized by LBFGS][out/mlp.png]

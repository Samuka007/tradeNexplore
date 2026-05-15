#import "@preview/cetz:0.5.0"
#let diagram = cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Tier 1: central question
  rect((-3.2, 6.8), (3.2, 7.5), fill: rgb("#e8e8e8"), stroke: 0.8pt, radius: 3pt, name: "question")
  content((0, 7.1), text(size: 10.5pt, weight: "bold")[What are you optimizing?], anchor: "center")

  // Arrows from question to categories
  line((-2.9, 6.75), (-2.9, 6.3), stroke: 0.8pt, mark: (end: ">"))
  line((2.9, 6.75), (2.9, 6.3), stroke: 0.8pt, mark: (end: ">"))

  // Tier 2: category boxes
  rect((-5, 5.6), (-0.8, 6.25), fill: rgb("#c3e6cb"), stroke: 0.8pt, radius: 3pt, name: "params")
  content((-2.9, 5.9), text(size: 9pt, weight: "bold")[Parameters], anchor: "center")

  rect((0.8, 5.6), (5, 6.25), fill: rgb("#f5c6cb"), stroke: 0.8pt, radius: 3pt, name: "struct")
  content((2.9, 5.9), text(size: 9pt, weight: "bold")[Structure], anchor: "center")

  // Arrows to algorithms
  line((-3.9, 5.55), (-3.9, 4.9), stroke: 0.7pt, mark: (end: ">"))
  line((-1.9, 5.55), (-1.9, 4.9), stroke: 0.7pt, mark: (end: ">"))
  line((2.9, 5.55), (2.9, 4.9), stroke: 0.7pt, mark: (end: ">"))

  // PSO
  rect((-5, 3.4), (-0.8, 4.85), fill: rgb("#d4edda"), stroke: 0.8pt, radius: 3pt, name: "pso")
  content((-2.9, 4.45), text(weight: "bold", size: 9.5pt)[PSO], anchor: "center")
  content((-2.9, 4.05), text(size: 8pt)[Swarm Intelligence], anchor: "center")
  content((-2.9, 3.72), text(size: 7.5pt, fill: rgb("#555555"))[Velocity update], anchor: "center")

  // ABC
  rect((-5, 1.3), (-0.8, 2.9), fill: rgb("#d4edda"), stroke: 0.8pt, radius: 3pt, name: "abc")
  content((-2.9, 2.45), text(weight: "bold", size: 9.5pt)[ABC], anchor: "center")
  content((-2.9, 2.05), text(size: 8pt)[Swarm Intelligence], anchor: "center")
  content((-2.9, 1.65), text(size: 7.5pt, fill: rgb("#555555"))[Role-based foraging], anchor: "center")

  // GP
  rect((0.8, 1.3), (5, 4.85), fill: rgb("#f8d7da"), stroke: 0.8pt, radius: 3pt, name: "gp")
  content((2.9, 4.45), text(weight: "bold", size: 9.5pt)[GP], anchor: "center")
  content((2.9, 3.95), text(size: 8pt)[Evolutionary], anchor: "center")
  content((2.9, 3.45), text(size: 7.5pt)[Program trees / Crossover], anchor: "center")
  content((2.9, 2.0), text(size: 7.5pt, fill: rgb("#842029"), weight: "bold")[Overfitting risk], anchor: "center")

  // HS baseline
  rect((-2, -0.4), (2, 0.9), fill: rgb("#cfe2ff"), stroke: 0.8pt, radius: 3pt, name: "hs")
  content((0, 0.55), text(weight: "bold", size: 9.5pt)[HS], anchor: "center")
  content((0, 0.1), text(size: 8pt)[Physical metaphor], anchor: "center")
  content((0, -0.15), text(size: 7.5pt, fill: rgb("#555555"))[Single-state baseline], anchor: "center")

  // Dashed lines from algorithms to HS baseline
  line((-2.9, 1.25), (-0.5, 0.95), stroke: (dash: "dashed", thickness: 0.6pt, paint: rgb("#666666")))
  line((2.9, 1.25), (0.5, 0.95), stroke: (dash: "dashed", thickness: 0.6pt, paint: rgb("#666666")))

  // Side annotations
  content((-5.3, 3.0), text(size: 8pt, weight: "bold")[Fix template,], anchor: "east")
  content((-5.3, 2.5), text(size: 8pt, weight: "bold")[tune values], anchor: "east")
  content((5.3, 3.0), text(size: 8pt, weight: "bold")[Discover template], anchor: "west")
})

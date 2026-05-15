#import "@preview/wordometer:0.1.5": total-words, word-count

#let logo-path = "assets/uwa-university-perth-logo-svg-vector.svg"

#let cover(
  title: (),
  subtitle: (),
  // students: (),
  department: "Department of Engineering and Mathematical Sciences",
  university: "The University of Western Australia",
  date: (),
) = {
  box[
    #align(center + top)[
      #text(size: 14pt, style: "italic")[#subtitle]
    ]

    #v(1em)

    #grid(
      columns: (25fr, 70fr),
      align: center + horizon,
      gutter: 0.5cm,
      [
        #image(logo-path)
      ],
      [
        #align(center)[
          #text(size: 22pt, weight: "bold")[#title]
          #v(-1em)
          #text(size: 12pt)[*Group 21* - #date - Total Pages: #context { counter(page).final().first() } ]
        ]
      ],
    )
  ]
}

(declare-project
  :name "krystal-janet-engine"
  :description "Janet Lisp Subproject Port of Krystal-Stack: Symplectic Hamiltonian Organism, Neural ASCII Raymarching Engine, Topological VM & Procedural OpenWorld"
  :author "Krystal-Stack Architecture Team"
  :dependencies ["https://github.com/janet-lang/spork.git"])

(declare-source
  :source ["krystal_sdf.janet"
           "cyclic_organism.janet"
           "neural_ascii_engine.janet"
           "governor.janet"
           "topological_vm.janet"
           "antigravity_peg.janet"
           "openworld_chunk.janet"
           "main.janet"])

(declare-executable
  :name "krystal-janet-engine"
  :entry "main.janet")

(declare-executable
  :name "krystal-janet-openworld"
  :entry "openworld_chunk.janet")

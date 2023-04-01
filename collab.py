######################################################################
# SETUP RF DIFFUSION
######################################################################

#@title setup **RFdiffusion** (~2m30S)
%%time
import os, time
if not os.path.isdir("params"):
  os.system("apt-get install aria2")
  os.system("mkdir params")
  # send param download into background
  os.system("(\
  aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt; \
  aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt; \
  aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar; \
  tar -xf alphafold_params_2022-12-06.tar -C params; \
  touch params/done.txt) &")

if not os.path.isdir("RFdiffusion"):
  print("installing RFdiffusion...")
  os.system("git clone https://github.com/sokrypton/RFdiffusion.git")
  os.system("pip -q install jedi omegaconf hydra-core icecream")
  os.system("pip -q install dgl -f https://data.dgl.ai/wheels/cu117/repo.html")
  os.system("cd RFdiffusion/env/SE3Transformer; pip -q install --no-cache-dir -r requirements.txt; pip -q install .")

if not os.path.isdir("colabdesign"):
  print("installing ColabDesign...")
  os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
  os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")

if not os.path.isdir("RFdiffusion/models"):
  print("downloading RFdiffusion params...")
  os.system("mkdir RFdiffusion/models")
  models = ["Base_ckpt.pt","Complex_base_ckpt.pt"]
  for m in models:
    while os.path.isfile(f"{m}.aria2"):
      time.sleep(5)
  os.system(f"mv {' '.join(models)} RFdiffusion/models")

import sys, random, string, re
if 'RFdiffusion' not in sys.path:
  os.environ["DGLBACKEND"] = "pytorch"
  sys.path.append('RFdiffusion')

from google.colab import files
from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb
from inference.utils import parse_pdb

def get_pdb(pdb_code=None):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"
  else:
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"

def run_diffusion(contigs, path, pdb=None, iterations=50,
                  symmetry="cyclic", copies=1, hotspot=None):
  # determine mode
  contigs = contigs.replace(","," ").replace(":"," ").split()
  is_fixed, is_free = False, False
  for contig in contigs:
    for x in contig.split("/"):
      a = x.split("-")[0]
      if a[0].isalpha():
        is_fixed = True
      if a.isnumeric():
        is_free = True
  if len(contigs) == 0 or not is_free:
    mode = "partial"
  elif is_fixed:
    mode = "fixed"
  else:
    mode = "free"

  # fix input contigs
  if mode in ["partial","fixed"]:
    pdb_filename = get_pdb(pdb)
    parsed_pdb = parse_pdb(pdb_filename)
    opts = f" inference.input_pdb={pdb_filename}"
    if mode in ["partial"]:
      partial_T = int(80 * (iterations / 200))
      opts += f" diffuser.partial_T={partial_T}"
      contigs = fix_partial_contigs(contigs, parsed_pdb)
    else:
      opts += f" diffuser.T={iterations}"
      contigs = fix_contigs(contigs, parsed_pdb)
  else:
    opts = f" diffuser.T={iterations}"
    parsed_pdb = None  
    contigs = fix_contigs(contigs, parsed_pdb)

  if hotspot is not None and hotspot != "":
    opts += f" ppi.hotspot_res=[{hotspot}]"

  # setup symmetry
  if copies > 1:
    sym = {"cyclic":"c","dihedral":"d"}[symmetry] + str(copies)
    sym_opts = f"--config-name symmetry  inference.symmetry={sym} \
    'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]' \
    potentials.olig_intra_all=True potentials.olig_inter_all=True \
    potentials.guide_scale=2 potentials.guide_decay=quadratic"
    opts = f"{sym_opts} {opts}"
    if symmetry == "dihedral": copies *= 2
    contigs = sum([contigs] * copies,[])

  opts = f"{opts} 'contigmap.contigs=[{' '.join(contigs)}]'"

  print("mode:", mode)
  print("output:", f"outputs/{path}")
  print("contigs:", contigs)

  cmd = f"./RFdiffusion/run_inference.py {opts} inference.output_prefix=outputs/{path} inference.num_designs=1"
  print(cmd)
  !{cmd}

  # fix pdbs
  pdbs = [f"outputs/traj/{path}_0_pX0_traj.pdb",
          f"outputs/traj/{path}_0_Xt-1_traj.pdb",
          f"outputs/{path}_0.pdb"]
  for pdb in pdbs:
    with open(pdb,"r") as handle: pdb_str = handle.read()
    with open(pdb,"w") as handle: handle.write(fix_pdb(pdb_str, contigs))
  return contigs, copies

######################################################################
# RUN RF DIFFUSION TO GENERATE A BACKBONE
######################################################################

#@title run **RFdiffusion** to generate a backbone
name = "test" #@param {type:"string"}
contigs = "100" #@param {type:"string"}
pdb = "" #@param {type:"string"}
copies = 1 #@param ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] {type:"raw"}
#@markdown ---
#@markdown **advanced** settings
iterations = 50 #@param ["50", "100", "150", "200"] {type:"raw"}
symmetry = "cyclic" #@param ["cyclic", "dihedral"]
hotspot = "" #@param {type:"string"}

# determine where to save
path = name
while os.path.exists(f"outputs/{path}_0.pdb"):
  path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

flags = {"contigs":contigs,
         "pdb":pdb,
         "copies":copies,
         "iterations":iterations,
         "symmetry":symmetry,
         "hotspot":hotspot,
         "path":path}

for k,v in flags.items():
  if isinstance(v,str):
    flags[k] = v.replace("'","").replace('"','')
         
contigs, copies = run_diffusion(**flags)

######################################################################
# DISPLAY A 3D STRUCTURE
######################################################################

#@title Display 3D structure {run: "auto"}
import py3Dmol
from colabdesign.shared.plot import pymol_color_list

from string import ascii_uppercase,ascii_lowercase
alphabet_list = list(ascii_uppercase+ascii_lowercase)

show_mainchains = False 
animate = False #@param {type:"boolean"}
color = "chain" #@param ["rainbow", "chain"]
hbondCutoff = 4.0
view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')

if animate:
  pdb = f"/content/outputs/traj/{path}_0_pX0_traj.pdb"
  pdb_str = open(pdb,'r').read()
  view.addModelsAsFrames(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
else:
  pdb = f"/content/outputs/{path}_0.pdb"
  pdb_str = open(pdb,'r').read()
  view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})

if color == "rainbow":
  view.setStyle({'cartoon': {'color':'spectrum'}})
elif color == "chain":
  for n,chain,color in zip(range(len(contigs)),
                           alphabet_list,
                           pymol_color_list):
      view.setStyle({'chain':chain},{'cartoon': {'color':color}})

view.zoomTo()
if animate:
  view.animate({'loop': 'backAndForth'})
view.show()


# run ProteinMPNN to generate a sequence and AlphaFold to validate
%%time
#@title run **ProteinMPNN** to generate a sequence and **AlphaFold** to validate
num_seqs = 8 #@param ["8", "16", "32", "64"] {type:"raw"}
initial_guess = False #@param {type:"boolean"}
num_recycles = 1 #@param ["0", "1", "2", "3", "6", "12"] {type:"raw"}
use_multimer = False #@param {type:"boolean"}
#@markdown - for **binder** design, we recommend `initial_guess=True num_recycles=3`

if not os.path.isfile("params/done.txt"):
  print("downloading AlphaFold params...")
  while not os.path.isfile("params/done.txt"):
    time.sleep(5)

contigs_str = ":".join(contigs)
opts = [f"--pdb=outputs/{path}_0.pdb",
        f"--loc=outputs/{path}",
        f"--contig={contigs_str}",
        f"--copies={copies}",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}"]
if initial_guess: opts.append("--initial_guess")
if use_multimer: opts.append("--use_multimer")
opts = ' '.join(opts)
!python colabdesign/rf/designability_test.py {opts}

######################################################################
# DISPLAY THE BEST RESULT
######################################################################

#@title Display best result
import py3Dmol
hbondCutoff = 4.0
view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')

pdb_str = open(f"outputs/{path}_0.pdb",'r').read()
view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
pdb_str = open(f"outputs/{path}/best.pdb",'r').read()
view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})

view.setStyle({"model":0},{'cartoon':{}}) #: {'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':100}}})
view.setStyle({"model":1},{'cartoon':{'colorscheme': {'prop':'b','gradient': 'roygb','min':0,'max':100}}})
view.zoomTo()
view.show()

######################################################################
# PACKAGE AND DOWNLOAD RESULTS
######################################################################

#@title Package and download results
#@markdown If you are having issues downloading the result archive, 
#@markdown try disabling your adblocker and run this cell again. 
#@markdown  If that fails click on the little folder icon to the 
#@markdown  left, navigate to file: `name.result.zip`, 
#@markdown  right-click and select \"Download\" 
#@markdown (see [screenshot](https://pbs.twimg.com/media/E6wRW2lWUAEOuoe?format=jpg&name=small)).
!zip -r {path}.result.zip outputs/{path}* outputs/traj/{path}*
files.download(f"{path}.result.zip")

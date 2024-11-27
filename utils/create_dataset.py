# (1) Look for all files that end with FGLn and FSLn in a folder
# then of course you will need to match them based on the number in the beginning

# (2) once you get all the files, simply use stitcher to stitch the off grounds (w/ 0 overlap)
# and (3) fuse them together with the ground (prob simple write to new file of all points, not overlapping)

# (4) class rebalancing and trai/eval split: you dont want cars to be too under-represented so you downsample 
# the most represented classes. Then you split. *All this is probably better to do on csv instead of las

# (5) eventually you can choose to reate a 'big' dataset where you downsample less. Then if gpu implementation works
# we might be able to use that one. But maybe it's not even necessary.
import multiprocessing
import random

import objaverse
import trimesh

random.seed(42)

# uids: list = objaverse.load_uids()  # Length: 798759

# annotations = objaverse.load_annotations()

# Example annotation:
# {'uri': 'https://api.sketchfab.com/v3/models/8476c4170df24cf5bbe6967222d1a42d',
#  'uid': '8476c4170df24cf5bbe6967222d1a42d',
#  'name': 'Iain_Dawson_Kew_Road_Formby',
#  'staffpickedAt': None,
#  'viewCount': 4,
#  'likeCount': 0,
#  'animationCount': 0,
#  'viewerUrl': 'https://sketchfab.com/3d-models/8476c4170df24cf5bbe6967222d1a42d',
#  'embedUrl': 'https://sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/embed',
#  'commentCount': 0,
#  'isDownloadable': True,
#  'publishedAt': '2021-03-18T09:36:25.430631',
#  'tags': [{'name': 'stair',
#    'slug': 'stair',
#    'uri': 'https://api.sketchfab.com/v3/tags/stair'},
#   {'name': 'staircase',
#    'slug': 'staircase',
#    'uri': 'https://api.sketchfab.com/v3/tags/staircase'},
#   {'name': 'staircon',
#    'slug': 'staircon',
#    'uri': 'https://api.sketchfab.com/v3/tags/staircon'}],
#  'categories': [],
#  'thumbnails': {'images': [{'uid': '606cf3aaaea14bb598913e803c7b26af',
#     'size': 37800,
#     'width': 1920,
#     'url': 'https://media.sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/thumbnails/03709bf568c34654b3ce1913fbf5bd2c/298948d530db40f1b783905cb20edb5d.jpeg',
#     'height': 1080},
#    {'uid': '143cb32654f14656b689b0ad1bd50d1b',
#     'size': 11662,
#     'width': 1024,
#     'url': 'https://media.sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/thumbnails/03709bf568c34654b3ce1913fbf5bd2c/ffa34231b88440a6957f41c4e1c919ec.jpeg',
#     'height': 576},
#    {'uid': '91fd98a0685044fb81900e1fcf2b047c',
#     'size': 6939,
#     'width': 720,
#     'url': 'https://media.sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/thumbnails/03709bf568c34654b3ce1913fbf5bd2c/180f2238f5e3444bb8362306b14c2ecc.jpeg',
#     'height': 405},
#    {'uid': 'f6219b24d9ff4961b52bc831f20093ee',
#     'size': 1756,
#     'width': 256,
#     'url': 'https://media.sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/thumbnails/03709bf568c34654b3ce1913fbf5bd2c/366be192697549b7b73943f8c01fc100.jpeg',
#     'height': 144},
#    {'uid': '568c9a813b6a4131a96aaf9c038e6c70',
#     'size': 550,
#     'width': 64,
#     'url': 'https://media.sketchfab.com/models/8476c4170df24cf5bbe6967222d1a42d/thumbnails/03709bf568c34654b3ce1913fbf5bd2c/108ea805ed214721acc180c51d1778f8.jpeg',
#     'height': 36}]},
#  'user': {'uid': 'b50b409d636f4a8e9af2111d370786bf',
#   'username': 'agrice',
#   'displayName': 'Alan Grice Staircase Co Ltd',
#   'profileUrl': 'https://sketchfab.com/agrice',
#   'account': 'basic',
#   'avatar': {'uri': 'https://api.sketchfab.com/v3/avatars/d17386e905724bd7a5384abd5d2f21e3',
#    'images': [{'size': 746,
#      'width': 32,
#      'url': 'https://media.sketchfab.com/avatars/d17386e905724bd7a5384abd5d2f21e3/339f3f93942b46b5a6ac44217789e298.jpeg',
#      'height': 32},
#     {'size': 1197,
#      'width': 48,
#      'url': 'https://media.sketchfab.com/avatars/d17386e905724bd7a5384abd5d2f21e3/6c1955e500b147298447f01354b1a0de.jpeg',
#      'height': 48},
#     {'size': 3174,
#      'width': 90,
#      'url': 'https://media.sketchfab.com/avatars/d17386e905724bd7a5384abd5d2f21e3/48fd73eced234a9ba6f94a251d53ad9b.jpeg',
#      'height': 90},
#     {'size': 3699,
#      'width': 100,
#      'url': 'https://media.sketchfab.com/avatars/d17386e905724bd7a5384abd5d2f21e3/9d25068a140f4c60b7e55cb63bd7860f.jpeg',
#      'height': 100}]},
#   'uri': 'https://api.sketchfab.com/v3/users/b50b409d636f4a8e9af2111d370786bf'},
#  'description': 'http://staircon.com/ <br>Export by <b>Alan Grice Staircase Co. Ltd</b> (lic 6391)',
#  'faceCount': 14608,
#  'createdAt': '2021-03-18T09:31:52.190927',
#  'vertexCount': 7309,
#  'isAgeRestricted': False,
#  'archives': {'glb': {'textureCount': 6,
#    'size': 1546940,
#    'type': 'glb',
#    'textureMaxResolution': 1024,
#    'faceCount': 14337,
#    'vertexCount': 19008},
#   'gltf': {'textureCount': 6,
#    'size': 932008,
#    'type': 'gltf',
#    'textureMaxResolution': 1024,
#    'faceCount': 14337,
#    'vertexCount': 19008},
#   'source': {'textureCount': None,
#    'size': 1026620,
#    'type': 'source',
#    'textureMaxResolution': None,
#    'faceCount': None,
#    'vertexCount': None},
#   'usdz': {'textureCount': None,
#    'size': 1373035,
#    'type': 'usdz',
#    'textureMaxResolution': None,
#    'faceCount': None,
#    'vertexCount': None}},
#  'license': 'by'}


# cc_by_uids = [
#     uid for uid, annotation in annotations.items() if annotation["license"] == "by"
# ]
# cc_by_uids[:10]

lvis_annotations = objaverse.load_lvis_annotations()
uids_to_load = lvis_annotations["tomato"][:10]

processes = multiprocessing.cpu_count()
objects = objaverse.load_objects(uids=uids_to_load, download_processes=processes)

trimesh.load(next(iter(objects.values()))).show()

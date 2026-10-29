"""TCN causale, GPS dense/creux ou baseline independante, sorties A/B/C."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv

from .graphes import Graphe
from .sources import G, Mo


class Temporel(nn.Module):
    def __init__(self, entrees: int, largeur: int):
        super().__init__()
        self.couches = nn.ModuleList([
            nn.Conv1d(entrees if i == 0 else largeur, largeur, 3, dilation=d)
            for i, d in enumerate((1, 2, 4))
        ])

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        h = x.permute(0, 2, 3, 1).reshape(b * n, f, t)
        for c in self.couches:
            h = F.relu(c(F.pad(h, (2 * c.dilation[0], 0))))
        return h.reshape(b, n, -1, t).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequence(x)[:, -1]


class BlocGPS(nn.Module):
    def __init__(self, largeur: int, tetes: int, attention: str):
        super().__init__()
        if attention not in ("dense", "creuse"):
            raise ValueError(f"Attention inconnue : {attention}")
        self.attention, self.tetes = attention, tetes
        self.local = GINEConv(nn.Sequential(nn.Linear(largeur, largeur), nn.ReLU(),
                                           nn.Linear(largeur, largeur)), edge_dim=10)
        if attention == "creuse":
            self.global_ = Mo.AttentionExphormer(largeur, tetes, 10)
        else:
            self.qkv = nn.Linear(largeur, 3 * largeur, bias=False)
        self.normes = nn.ModuleList([nn.LayerNorm(largeur) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(largeur, 2 * largeur), nn.ReLU(),
                                 nn.Linear(2 * largeur, largeur))

    def forward(self, h, local, alocal, attention, aattention, masque):
        b, n, w = h.shape
        plat = h.reshape(b * n, w)
        hl = self.normes[0](plat + self.local(plat, local, alocal))
        if self.attention == "creuse":
            hg = self.global_(plat, attention, aattention, masque)
        else:
            q, k, v = self.qkv(h).reshape(b, n, 3, self.tetes, w // self.tetes).permute(
                2, 0, 3, 1, 4).unbind(0)
            hg = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b * n, w)
        fusion = hl + self.normes[1](plat + hg)
        return self.normes[2](fusion + self.ffn(fusion)).reshape(b, n, w)


class BlocIndependant(nn.Module):
    def __init__(self, largeur: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(largeur, 2 * largeur), nn.ReLU(),
                                 nn.Linear(2 * largeur, largeur))
        self.norme = nn.LayerNorm(largeur)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norme(h + self.ffn(h))


class Prototype(nn.Module):
    def __init__(self, g: Graphe, representation: str, attention: str,
                 entrees: int = 6, largeur: int = 64, profondeur: int = 4,
                 entrees_liens: int = 0, statiques: torch.Tensor | None = None,
                 statiques_liens: torch.Tensor | None = None,
                 architecture: str = "gps"):
        super().__init__()
        if representation not in ("line", "hydraulique"):
            raise ValueError(f"Representation inconnue : {representation}")
        if architecture not in ("gps", "independant"):
            raise ValueError(f"Architecture inconnue : {architecture}")
        if architecture == "independant" and (representation != "line" or entrees_liens):
            raise ValueError("La baseline independante utilise les entrees fusionnees line uniquement")
        self.architecture = architecture
        self.representation = representation
        self.n = len(g.noms)
        self.register_buffer("local", g.aretes)
        self.register_buffer("alocal", g.attributs)
        # Les deux branches ont le meme sommet global. Le creux est une union sans doublons.
        if architecture == "gps":
            att, aa = G.augmenter(g.aretes, g.attributs, self.n, degre=6, graine=9)
            codes = att[0] * (self.n + 1) + att[1]
            ordre = torch.argsort(codes, stable=True)
            garde = torch.ones(len(ordre), dtype=torch.bool)
            garde[1:] = codes[ordre[1:]] != codes[ordre[:-1]]
            ordre = ordre[garde]
            self.register_buffer("att", att[:, ordre])
            self.register_buffer("aa", aa[ordre])
        else:
            self.register_buffer("att", torch.empty((2, 0), dtype=torch.long))
            self.register_buffer("aa", torch.empty((0, 10)))
        self.register_buffer("segments", g.segments)
        self.register_buffer("cibles_fuite", g.cibles_fuite)
        self.temporel = Temporel(entrees, largeur)
        self.temporel_liens = Temporel(entrees_liens, largeur) if entrees_liens else None
        if self.temporel_liens is not None:
            if representation != "hydraulique":
                raise ValueError("Le line graph recoit deja les liens dans ses entrees")
            self.depuis_debut = nn.Linear(largeur, largeur, bias=False)
            self.depuis_fin = nn.Linear(largeur, largeur, bias=False)
        if statiques is not None:
            if statiques.ndim != 2 or statiques.shape[0] != self.n:
                raise ValueError("Attributs statiques incompatibles avec le graphe")
            self.register_buffer("statiques", statiques.float())
            self.projection_statique = nn.Linear(statiques.shape[1], largeur)
        else:
            self.register_buffer("statiques", None)
        if statiques_liens is not None:
            if self.temporel_liens is None or statiques_liens.ndim != 2 \
                    or statiques_liens.shape[0] != self.local.shape[1] // 2:
                raise ValueError("Attributs statiques de liens incompatibles")
            self.register_buffer("statiques_liens", statiques_liens.float())
            self.projection_statique_liens = nn.Linear(statiques_liens.shape[1], largeur)
        else:
            self.register_buffer("statiques_liens", None)
        self.virtuel = nn.Parameter(torch.zeros(1, 1, largeur)) if architecture == "gps" else None
        self.blocs = nn.ModuleList([
            BlocGPS(largeur, 4, attention) if architecture == "gps" else BlocIndependant(largeur)
            for _ in range(profondeur)])
        self.q = nn.Linear(largeur if representation == "line" else 2 * largeur, 1)
        self.fuite = nn.Linear(2 * largeur if representation == "line" else largeur, 1)
        self._lots: dict = {}

    def _lot(self, b: int):
        cle = (b, str(self.local.device))
        if cle not in self._lots:
            e, a = Mo.empiler(self.local, self.alocal, self.n + 1, b)
            att, aa = Mo.empiler(self.att, self.aa, self.n + 1, b)
            self._lots[cle] = (e, a, att, aa, att[1] % (self.n + 1) == self.n)
        return self._lots[cle]

    def forward(self, x: torch.Tensor, objectif: str = "C",
                liens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if objectif not in ("A", "B", "C"):
            raise ValueError(f"Objectif inconnu : {objectif}")
        b = x.shape[0]
        h = self.temporel(x)
        if self.statiques is not None:
            h = h + self.projection_statique(self.statiques)[None]
        if self.temporel_liens is not None:
            if liens is None or liens.shape[:2] != x.shape[:2] \
                    or liens.shape[2] != self.local.shape[1] // 2:
                raise ValueError("Historiques de liens absents ou incompatibles")
            hl = self.temporel_liens(liens)
            if self.statiques_liens is not None:
                hl = hl + self.projection_statique_liens(self.statiques_liens)[None]
            ext = self.local[:, ::2]
            for indices, projection in ((ext[0], self.depuis_debut),
                                        (ext[1], self.depuis_fin)):
                messages = projection(hl)
                somme = torch.zeros_like(h).index_add(1, indices, messages)
                degre = torch.bincount(indices, minlength=self.n).clamp_min(1)
                h = h + somme / degre[None, :, None]
        elif liens is not None:
            raise ValueError("Ce modele ne possede pas d'encodeur de liens")
        if self.architecture == "gps":
            h = torch.cat((h, self.virtuel.expand(b, -1, -1)), 1)
            lot = self._lot(b)
            for bloc in self.blocs:
                h = bloc(h, *lot)
        else:
            for bloc in self.blocs:
                h = bloc(h)
        sorties = {}
        if objectif in ("A", "C"):
            if self.representation == "line":
                q = self.q(h[:, self.segments]).squeeze(-1)
            else:
                # Une paire dirigee sur deux dans le graphe hydraulique.
                ext = self.local[:, ::2].T[self.segments]
                q = self.q(torch.cat((h[:, ext[..., 0]], h[:, ext[..., 1]]), -1)).squeeze(-1)
            sorties["q_m3h_normalise"] = q
        if objectif in ("B", "C"):
            if self.representation == "line":
                paire = h[:, self.cibles_fuite]
                z = torch.cat((paire[:, :, 0], paire[:, :, 1]), -1)
            else:
                z = h[:, self.cibles_fuite]
            sorties["fuite_logits"] = self.fuite(z).squeeze(-1)
        return sorties

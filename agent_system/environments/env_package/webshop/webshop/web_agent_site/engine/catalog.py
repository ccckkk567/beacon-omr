"""Static catalog construction; no browser, JVM, session or live model handles."""
import os
import random

import numpy as np

from .engine import load_products
from .goal import get_goals
from web_agent_site.utils import random_idx, HUMAN_ATTR_PATH


def catalog_binding(seed, file_path, attr_path, num_products=None, human_goals=0,
                    limit_goals=-1, human_attr_path=None):
    return dict(seed=seed, file_path=os.path.abspath(file_path), attr_path=os.path.abspath(attr_path),
                human_attr_path=os.path.abspath(human_attr_path or HUMAN_ATTR_PATH),
                num_products=num_products, human_goals=bool(human_goals), limit_goals=limit_goals)


def load_catalog(seed, file_path, attr_path, filter_goals=None, limit_goals=-1,
                 num_products=None, human_goals=0, human_attr_path=None):
    products, items, prices, _ = load_products(file_path, attr_path, num_products=num_products,
        human_goals=human_goals, human_attr_path=human_attr_path)
    goals = get_goals(products, prices, human_goals)
    random.seed(seed)
    random.shuffle(goals)
    if filter_goals is not None:
        goals = [goal for i, goal in enumerate(goals) if filter_goals(i, goal)]
    if limit_goals != -1 and limit_goals < len(goals):
        cumulative = [0] + np.cumsum([goal['weight'] for goal in goals]).tolist()
        indices = []
        while len(indices) < limit_goals:
            index = random_idx(cumulative)
            if index not in indices:
                indices.append(index)
        goals = [goals[i] for i in indices]
    return dict(binding=catalog_binding(seed, file_path, attr_path, num_products, human_goals,
                limit_goals, human_attr_path), all_products=products, product_item_dict=items,
                product_prices=prices, goals=goals, random_state=random.getstate())

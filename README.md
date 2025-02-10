    tmp.next = 0;是头哨兵节点
    tmp.y0 = INT_MAX;edges.push_back(tmp);尾哨兵节点，所有y都是小于该tmp.y0，即保证不会有新增边，只会出边或者入边。
    出边：y到达边y1时，prelast->next = last->next删除旧边(last)，指向新边
    入边：last->x<e->x or y<e->y0, prelast = last, last = last->next,直接移动边
    增边：prelast->next=e;e->next=last;prelast=e;e=edges[++i] 指新边，新边连末边(0或者last)

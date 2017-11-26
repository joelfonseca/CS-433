"""Contains all the postprocessing functions used."""
def is_outlier(img, x, y, color, IMG_PATCH_SIZE):
    outlier = True
    for i in [-IMG_PATCH_SIZE, 0, IMG_PATCH_SIZE]:
        for j in [-IMG_PATCH_SIZE, 0, IMG_PATCH_SIZE]:
            if i is not j:
                #print('color ', img[x+i][y+j], ' at (', x+i, ', ', y+j, ')')
                if img[x+i][y+j] == color:
                    outlier = False
                    return outlier
        
    return outlier

def update_pixel_batch(img, x, y, opp_color):
    for i in range(0,16):
        for j in range(0,16):
            img[x+i][y+j] = opp_color

def delete_outlier(img, IMG_PATCH_SIZE):
    # img is of size 608x608
    img_size = img.shape[0]

    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
            color = img[i][j]
            opp_color = 0 if color==1 else 1
            if is_outlier(img, i, j, color, IMG_PATCH_SIZE):
                #print('one outlier found at: ', i, ' ', j)
                #print('color before: ', img[i][j])
                update_pixel_batch(img, i, j, opp_color)
                #print('color after: ', img[i][j], img[i+2][j+2])

def tetris_shape_cleaner(img, IMG_PATCH_SIZE):
    # img is of size 608x608
    img_size = img.shape[0]

    # size of filler
    filler_size = 3


    
    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):

            ROAD = 1
            OTHER = 0

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==OTHER and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
               ):
                #r r r       r r r
                #r o r   ->  r r r
                #o o o       o o o 
                
                print('found a _|_ at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)
            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==OTHER 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
               ):
                #r r o       r r o
                #r o o   ->  r r o
                #r r o       r r o

                print('found a _|_ (90) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==OTHER and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
               ):
                #o o o       o o o
                #r o r   ->  r r r
                #r r r       r r r

                print('found a _|_ (180) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
                    and img[i][j-IMG_PATCH_SIZE]==OTHER and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
               ):
                #o r r       o r r
                #o o r   ->  o r r
                #o r r       o r r

                print('found a _|_ (270) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

            """
            BAD
            if (    img[i-IMG_PATCH_SIZE][j]==ROAD and img[i][j]==OTHER and img[i+IMG_PATCH_SIZE][j]==ROAD
               ):

               #r      r
               #o  ->  r
               #r      r
               print('found : at: (', i, ',', j, ')')
               update_pixel_batch(img, i, j, ROAD)
            """

            # inverse
            ROAD = 0
            OTHER = 1
        

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==OTHER and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
               ):
                #r r r       r r r
                #r o r   ->  r r r
                #o o o       o o o 
                
                print('found inv a _|_ at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)
            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==OTHER 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
               ):
                #r r o       r r o
                #r o o   ->  r r o
                #r r o       r r o

                print('found inv a _|_ (90) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==OTHER and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                    and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==ROAD and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
               ):
                #o o o       o o o
                #r o r   ->  r r r
                #r r r       r r r

                print('found inv a _|_ (180) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

            if (    img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
                    and img[i][j-IMG_PATCH_SIZE]==OTHER and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD 
                    and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==ROAD
               ):
                #o r r       o r r
                #o o r   ->  o r r
                #o r r       o r r

                print('found inv a _|_ (270) at: (', i, ',', j, ')')
                update_pixel_batch(img, i, j, ROAD)

def border_cleaner(img, IMG_PATCH_SIZE):
    # img is of size 608x608
    img_size = img.shape[0]

    ROAD = 1
    OTHER = 0

    # check four corners of image
    if img[0][0]==ROAD and img[IMG_PATCH_SIZE][0]==OTHER and img[0][IMG_PATCH_SIZE]==OTHER and img[IMG_PATCH_SIZE][IMG_PATCH_SIZE]==OTHER:
        # top left corner
        update_pixel_batch(img, 0, 0, OTHER)
    if img[0][img_size-IMG_PATCH_SIZE]==ROAD and img[0][img_size-2*IMG_PATCH_SIZE]==OTHER and img[IMG_PATCH_SIZE][img_size-IMG_PATCH_SIZE]==OTHER and img[IMG_PATCH_SIZE][img_size-2*IMG_PATCH_SIZE]==OTHER:
        # top right corner
        update_pixel_batch(img, 0, img_size-IMG_PATCH_SIZE, OTHER)
    if img[img_size-IMG_PATCH_SIZE][0]==ROAD and img[img_size-IMG_PATCH_SIZE][IMG_PATCH_SIZE]==OTHER and img[img_size-2*IMG_PATCH_SIZE][0]==OTHER and img[img_size-2*IMG_PATCH_SIZE][IMG_PATCH_SIZE]==OTHER:
        # bottom left corner
        update_pixel_batch(img, img_size-IMG_PATCH_SIZE, 0, OTHER)
    if img[img_size-IMG_PATCH_SIZE][img_size-IMG_PATCH_SIZE]==ROAD and img[img_size-IMG_PATCH_SIZE][img_size-2*IMG_PATCH_SIZE]==OTHER and img[img_size-2*IMG_PATCH_SIZE][img_size-2*IMG_PATCH_SIZE]==OTHER and img[img_size-2*IMG_PATCH_SIZE][img_size-IMG_PATCH_SIZE]==OTHER:
        # bottom right corner
        update_pixel_batch(img, img_size-IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, OTHER)

    # check left border
    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        if (   img[i-IMG_PATCH_SIZE][0]==OTHER and img[i][0]==ROAD and img[i+IMG_PATCH_SIZE][0]==OTHER
               and img[i-IMG_PATCH_SIZE][IMG_PATCH_SIZE]==OTHER and img[i][IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][IMG_PATCH_SIZE]==OTHER
           ):
           update_pixel_batch(img, i, 0, OTHER)

    # check right border
    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        if (   img[i-IMG_PATCH_SIZE][img_size-2*IMG_PATCH_SIZE]==OTHER and img[i][img_size-2*IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][img_size-2*IMG_PATCH_SIZE]==OTHER
               and img[i-IMG_PATCH_SIZE][img_size-IMG_PATCH_SIZE]==OTHER and img[i][img_size-IMG_PATCH_SIZE]==ROAD and img[i+IMG_PATCH_SIZE][img_size-IMG_PATCH_SIZE]==OTHER
           ):
           update_pixel_batch(img, i, img_size-IMG_PATCH_SIZE, OTHER)

    # check bottom border
    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        if (   img[img_size-2*IMG_PATCH_SIZE][i-IMG_PATCH_SIZE]==OTHER and img[img_size-2*IMG_PATCH_SIZE][i]==OTHER and img[img_size-2*IMG_PATCH_SIZE][i+IMG_PATCH_SIZE]==OTHER
               and img[img_size-IMG_PATCH_SIZE][i-IMG_PATCH_SIZE]==OTHER and img[img_size-IMG_PATCH_SIZE][i]==ROAD and img[img_size-IMG_PATCH_SIZE][i+IMG_PATCH_SIZE]==OTHER
           ):
           update_pixel_batch(img, img_size-IMG_PATCH_SIZE, i, OTHER)
    
    # check top border
    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        if (   img[0][i-IMG_PATCH_SIZE]==OTHER and img[0][i]==ROAD and img[0][i+IMG_PATCH_SIZE]==OTHER
               and img[IMG_PATCH_SIZE][i-IMG_PATCH_SIZE]==OTHER and img[IMG_PATCH_SIZE][i]==OTHER and img[IMG_PATCH_SIZE][i+IMG_PATCH_SIZE]==OTHER
           ):
           update_pixel_batch(img, 0, i, OTHER)

def region_cleaner(img, IMG_PATCH_SIZE):
    # img is of size 608x608
    img_size = img.shape[0]

    ROAD = 1
    OTHER = 0

    #o o o o        o o o o
    #o r r o   ->   o o o o
    #o o o o        o o o o

    for i in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(IMG_PATCH_SIZE, img_size-2*IMG_PATCH_SIZE, IMG_PATCH_SIZE):

            if (   img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==OTHER and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j+2*IMG_PATCH_SIZE]==OTHER
                   and img[i][j-IMG_PATCH_SIZE]==OTHER and img[i][j]==ROAD and img[i][j+IMG_PATCH_SIZE]==ROAD and img[i][j+2*IMG_PATCH_SIZE]==OTHER
                   and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==OTHER and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j+2*IMG_PATCH_SIZE]==OTHER
               ):
               update_pixel_batch(img, i, j, OTHER)
               update_pixel_batch(img, i, j+IMG_PATCH_SIZE, OTHER)


    #o o o         o o o
    #o r o    ->   o o o
    #o r o         o o o
    #o o o         o o o

    for i in range(IMG_PATCH_SIZE, img_size-2*IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(IMG_PATCH_SIZE, img_size-IMG_PATCH_SIZE, IMG_PATCH_SIZE):

            if (   img[i-IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i-IMG_PATCH_SIZE][j]==OTHER and img[i-IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                   and img[i][j-IMG_PATCH_SIZE]==OTHER and img[i][j]==ROAD and img[i][j+IMG_PATCH_SIZE]==OTHER
                   and img[i+IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
                   and img[i+2*IMG_PATCH_SIZE][j-IMG_PATCH_SIZE]==OTHER and img[i+2*IMG_PATCH_SIZE][j]==OTHER and img[i+2*IMG_PATCH_SIZE][j+IMG_PATCH_SIZE]==OTHER
               ):
               update_pixel_batch(img, i, j, OTHER)
               update_pixel_batch(img, i+IMG_PATCH_SIZE, j, OTHER)

def naive_cleaner(img, IMG_PATCH_SIZE):
    # img is of size 608x608
    img_size = img.shape[0]

    ROAD = 1
    OTHER = 0

    #r r o r r   ->   r r r r r
    #very bad for one particular image

    for i in range(0, img_size, IMG_PATCH_SIZE):
        for j in range(2*IMG_PATCH_SIZE, img_size-2*IMG_PATCH_SIZE, IMG_PATCH_SIZE):

            if (   img[i][j-2*IMG_PATCH_SIZE]==ROAD and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==ROAD and img[i][j+2*IMG_PATCH_SIZE]==ROAD
                ):
                update_pixel_batch(img, i, j, ROAD)

    #r r r o o r r r   ->   r r r r r r r

    for i in range(0, img_size, IMG_PATCH_SIZE):
        for j in range(3*IMG_PATCH_SIZE, img_size-4*IMG_PATCH_SIZE, IMG_PATCH_SIZE):

            if (   img[i][j-3*IMG_PATCH_SIZE]==ROAD and img[i][j-2*IMG_PATCH_SIZE]==ROAD and img[i][j-IMG_PATCH_SIZE]==ROAD and img[i][j]==OTHER and img[i][j+IMG_PATCH_SIZE]==OTHER and img[i][j+2*IMG_PATCH_SIZE]==ROAD and img[i][j+3*IMG_PATCH_SIZE]==ROAD and img[i][j+4*IMG_PATCH_SIZE]==ROAD
                ):
                update_pixel_batch(img, i, j, ROAD)
                update_pixel_batch(img, i, j+IMG_PATCH_SIZE, ROAD)

    #r        r
    #r        r
    #o   ->   r
    #r        r
    #r        r

    for i in range(2*IMG_PATCH_SIZE, img_size-2*IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(0, img_size, IMG_PATCH_SIZE):

            if (   img[i-2*IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i][j]==OTHER and img[i+IMG_PATCH_SIZE][j]==ROAD and img[i+2*IMG_PATCH_SIZE][j]==ROAD
                ):
                update_pixel_batch(img, i, j, ROAD)

    #r        r
    #r        r
    #r        r
    #o   ->   r
    #o        r
    #r        r
    #r        r
    #r        r

    for i in range(3*IMG_PATCH_SIZE, img_size-4*IMG_PATCH_SIZE, IMG_PATCH_SIZE):
        for j in range(0, img_size, IMG_PATCH_SIZE):

            if (   img[i-3*IMG_PATCH_SIZE][j]==ROAD and img[i-2*IMG_PATCH_SIZE][j]==ROAD and img[i-IMG_PATCH_SIZE][j]==ROAD and img[i][j]==OTHER and img[i+IMG_PATCH_SIZE][j]==OTHER and img[i+2*IMG_PATCH_SIZE][j]==ROAD and img[i+3*IMG_PATCH_SIZE][j]==ROAD and img[i+4*IMG_PATCH_SIZE][j]==ROAD
                ):
                update_pixel_batch(img, i, j, ROAD)
                update_pixel_batch(img, i+IMG_PATCH_SIZE, j, ROAD)

import pygame
import random #Thu vien random dung de tao thuc an xuat hien ngau hien tren man hinh 

#Khoi tao game
pygame.init()

#Chieu rong chieu cao cua man hinh
width=800
height=600

#Thiet lap man hinh game va ten game
screen = pygame.display.set_mode((width,height))
pygame.display.set_caption('Snake game in Python - AH_NQ')

#Cac mau trong game
#Cac mau RGB co the tham khao tai RGB color selector
black = (13, 13, 12)
red = (255, 0, 0)
blue = (25, 17, 245)
BACKGROUND = (241, 245, 17)
white = (255,255,255)

#Bien clock xac dinh toc do choi game - do kho cua game
clock = pygame.time.Clock()
speed = 20 #Bien speed xac dinh toc do game
space = 10 #Khoang cach de con ran di chuyen

#Font chu trong game
fontStyle = pygame.font.SysFont(None,30)
score_font = pygame.font.SysFont("monospace", 25)

#Ham tra ve thong diep trong game
def message(msg, color):
    mess = fontStyle.render(msg,True,color)
    screen.blit(mess, (40,280))

#Ham hien thi con ran tren man hinh
def show_snake(space, snake_list):
    for j in snake_list:
        pygame.draw.rect(screen,black,[j[0],j[1],space,space])

#Ham hien thi so diem cua nguoi choi
def Your_score(score):
    value = score_font.render("Diem hien tai: " + str(score), True, red)
    screen.blit(value, [0, 0])

#Ham chinh cua chuong trinh 
def snake():

    #Bien over xac dinh nguoi choi thu cuoc -> Tro choi ket thuc
    #Bien close hoi nguoi choi co muon tiep tuc choi lai hay khong neu khong thi bien over duoc thuc thi

    game_over = False
    game_close = False

    #Xac dinh con ran nam o giua man hinh ngay tu ban dau    
    x = width/2
    y = height/2

    #Xac dinh vi tri sau khi con ran di chuyen
    x_change = 0
    y_change = 0

    #Tang do dai cho ran su dung LIST(List trong python co the chua nhieu kieu du lieu khac nhau)
    snake_List = []
    Length_of_snake = 1 #Do dai ban dau cua ran cung nhu la so diem cua nguoi choi tru di 1(Vi ban dau so diem bang 0)

    #Xac dinh thuc cho snake 
    #Random ra toa do(x,y) de thuc an hien thi len man hinh
    food_x = round(random.randrange(0, width - space) / 10.0) * 10.0
    food_y = round(random.randrange(0, height - space) / 10.0) * 10.0

    #Trong khi nguoi choi chua thua
    while not game_over:

        #Vong lap hoi nguoi choi dung lai hay choi tiep
        while game_close == True:
            screen.fill(white)
            message('THUA ROI !!!! Chon phim Q de thoat hoac chon phim C de tiep tuc tro choi',red)
            pygame.display.update()

            #Xu ly su kien khi nguoi choi chon tiep tuc hay dung lai
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: #QUIT
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c: #CONTINUE
                        snake() #Goi lai ham snake() de nguoi choi tiep tuc chuong trinh
                elif event.type == pygame.QUIT: #Neu nguoi dung an nut X - coi nhu thoat chuong trinh
                    game_over = True
                    game_close = False

        #XLSK khi nguoi choi di chuyen trong game
        for event in pygame.event.get():
            #XLSK khi nguoi choi bam thoat
            if event.type == pygame.QUIT:
               game_over = True

            #XLSK khi nguoi choi bam cac nut di chuyen
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -space
                    y_change = 0
                elif event.key == pygame.K_RIGHT:
                    x_change = space
                    y_change = 0
                elif event.key == pygame.K_UP:
                    y_change = -space
                    x_change = 0
                elif event.key == pygame.K_DOWN:
                    y_change = space
                    x_change = 0
                
        #XSLK khi nguoi choi di chuyen con ran dam vao man hinh
        if x >=width or x < 0 or y >= height or y < 0:
           game_close = True

        #Thay doi toa do snake khi nguoi choi di chuyen
        x += x_change 
        y += y_change

        #Hien thi background cua man hinh kem theo toa do cua thuc an
        screen.fill(BACKGROUND)
        pygame.draw.rect(screen,blue,[food_x,food_y,space,space])

        #Snakehead chua toa do cua x va y
        snake_Head = []
        snake_Head.append(x)
        snake_Head.append(y)

        #Danh sach chua toa do cua con ran danh sach Snake_List se chua danh sach snake_Head
        snake_List.append(snake_Head) 

        #Cau dieu kien xac dinh cho tro choi duoc thuc thi
        #Neu khong co cau lenh nay mac dinh con ran ban dau se la than cua no va no dang cham vao than cua no la pham qui
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        #Khi ran dung vao than cua no thi cho troi ket thuc
        for j in snake_List[:-1]:
            if j == snake_Head:
                game_close = True

        #Show Snake - Hien thi con ran ra man hinh
        show_snake(space,snake_List)
        Your_score(Length_of_snake - 1)

        #Cap nhat nhung thay doi trong game
        pygame.display.update()

        #neu Snake an moi thi tang diem cua nguoi choi len 1 don vi va dong thoi random ra 1 thuc an co vi tri moi trong man hinh
        if x == food_x and y == food_y:
            food_x = round(random.randrange(0, width - space) / 10.0) * 10.0
            food_y = round(random.randrange(0, height - space) / 10.0) * 10.0
            Length_of_snake += 1
        
        clock.tick(speed) #Do kho cua game        

    #Thoat tro choi va xoa du lieu tiet kiem bo nho
    pygame.quit()

#Goi ham snake
snake()
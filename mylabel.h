#ifndef MYLABEL_H
#define MYLABEL_H

#include <QLabel>
#include <QMouseEvent>
#include <QEvent>
#include <QApplication>
#include <QDebug>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

class MyLabel : public QLabel
{
    Q_OBJECT
public:
    explicit MyLabel(QWidget *parent = 0);    

    void mouseMoveEvent(QMouseEvent *ev);
    void mousePressEvent(QMouseEvent *ev);
    void mousePressEvent(QMouseEvent *ev,int id_);
    void mouseReleaseEvent(QMouseEvent *ev);
    void leaveEvent(QEvent *event);    
    void wheelEvent(QWheelEvent *event);

    void changeEvent(QEvent *ev);

    Qt::MouseButton MouseButtonPressed;
    Qt::KeyboardModifiers keyboardModifier;
    bool buttonPressed;

    int id;
    int x;
    int y;
    int prevX;
    int prevY;
    int xZoom;
    int yZoom;
    int xScaled;
    int yScaled;
    QPixmap srcOriginal;

    void scaleImage(double factor);

    void setImage(cv::Mat &image);
    void setImage(QPixmap image);

signals:
    void MousePressed(QMouseEvent *ev);
    void MousePressed(QMouseEvent *ev,int id_);
    void MouseWheel(QWheelEvent *event);
    void MouseWheel(QWheelEvent *event,int id_);
    void MousePos();
    void MousePos(QMouseEvent *ev,int id_);
    void MouseLeft();
    void MouseRelease();


public slots:


public:

    float m_scaleFactor;
  // float m_PreviousscaleFactor=1;

    cv::Mat actualImage;


    QImage Mat2QImage(const cv::Mat &src);
    int getId() const;
    void setId(int value);
    int getXZoom() const;
    void setXZoom(int value);
    int getYZoom() const;
    void setYZoom(int value);
    int getPrevX() const;
    void setPrevX(int value);
    int getPrevY() const;
    void setPrevY(int value);
    int getXScaled() const;
    void setXScaled(int value);
    int getYScaled() const;
    void setYScaled(int value);
    QPixmap getSrcOriginal() const;
    void setSrcOriginal(const QPixmap &value);
};

#endif // MYLABEL_H

#include "mylabel.h"
#define DEFAULT_ZOOM_FACTOR 1.2
#define DEFAULT_ZOOM_CTRL_FACTOR 1.02

MyLabel::MyLabel(QWidget *parent) : QLabel(parent){
    m_scaleFactor = 1.0;
    setMouseTracking(true);
}

void MyLabel::mouseMoveEvent(QMouseEvent *ev){
    if(pixmap() == 0){
        return;
    }
    this->prevX = this->x;
    this->prevY = this->y;
    this->x = ev->x();
    this->y = ev->y();
    buttonPressed = ev->buttons() & Qt::LeftButton;
    emit MousePos(ev,this->id);
}

void MyLabel::mousePressEvent(QMouseEvent *ev)
{
    if(pixmap() == 0){
        return;
    }
    this->x = ev->x();
    this->y = ev->y();
    MouseButtonPressed = ev->button();
    keyboardModifier   = ev->modifiers();
    emit MousePressed(ev,this->id);
}

void MyLabel::mousePressEvent(QMouseEvent *ev,int id_)
{
    if(pixmap() == 0){
        return;
    }
    this->x = ev->x();
    this->y = ev->y();
    MouseButtonPressed = ev->button();
    keyboardModifier   = ev->modifiers();
    emit MousePressed(ev,this->id);
}

void MyLabel::mouseReleaseEvent(QMouseEvent *ev)
{
    Q_UNUSED(ev)
    emit MouseRelease();
}

void MyLabel::leaveEvent(QEvent *event)
{
    Q_UNUSED(event)
    emit MouseLeft();
}

void MyLabel::wheelEvent(QWheelEvent *event)
{
    if(pixmap() == 0){
        return;
    }
    //emit MouseWheel(event);
    emit MouseWheel(event,this->id);
    if((event->modifiers() & Qt::ShiftModifier) || event->modifiers() & Qt::ControlModifier)
        return;

 //   double factor = (event->modifiers() & Qt::ControlModifier) ? DEFAULT_ZOOM_CTRL_FACTOR : DEFAULT_ZOOM_FACTOR;
     double factor = DEFAULT_ZOOM_FACTOR;

    //m_PreviousscaleFactor=m_scaleFactor;

    xZoom = event->x();
    yZoom = event->y();

    if(event->delta() > 0){
        // Zoom in
        m_scaleFactor *= factor;
    }
    else{
        m_scaleFactor=1;
    }
    //scaleImage(m_scaleFactor);
    //std::cout << m_scaleFactor << std::endl;
}

void MyLabel::changeEvent(QEvent *ev)
{
    Q_UNUSED(ev)
//    if(pixmap() != 0)
//        scaleImage(m_scaleFactor);
}

QPixmap MyLabel::getSrcOriginal() const
{
    return srcOriginal;
}

void MyLabel::setSrcOriginal(const QPixmap &value)
{
    srcOriginal = value;
}

int MyLabel::getYScaled() const
{
    return yScaled;
}

void MyLabel::setYScaled(int value)
{
    yScaled = value;
}

int MyLabel::getXScaled() const
{
    return xScaled;
}

void MyLabel::setXScaled(int value)
{
    xScaled = value;
}

int MyLabel::getPrevY() const
{
    return prevY;
}

void MyLabel::setPrevY(int value)
{
    prevY = value;
}

int MyLabel::getPrevX() const
{
    return prevX;
}

void MyLabel::setPrevX(int value)
{
    prevX = value;
}

int MyLabel::getYZoom() const
{
    return yZoom;
}

void MyLabel::setYZoom(int value)
{
    yZoom = value;
}

int MyLabel::getXZoom() const
{
    return xZoom;
}

void MyLabel::setXZoom(int value)
{
    xZoom = value;
}

int MyLabel::getId() const
{
    return id;
}

void MyLabel::setId(int value)
{
    id = value;
}

void MyLabel::scaleImage(double factor) {
    Q_UNUSED(factor)
    if(pixmap() == 0){
        return;
    }
    if(factor == 1.0){
        this->setPixmap(this->srcOriginal);
        return;
    }

    QSize size = pixmap()->size() * m_scaleFactor;
    QPixmap aux = pixmap()->scaled(size, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    int xaux = size.width() * xZoom / pixmap()->width();
    int yaux = size.height() * yZoom / pixmap()->height();

    QPoint topleft;

    int left = std::max(xaux - pixmap()->size().width()/2, 0);
    int top = std::max(yaux - pixmap()->size().height()/2, 0);

    if(left > size.width() - pixmap()->size().width()){
        left = size.width() - pixmap()->size().width();
    }

    if(top > size.height() - pixmap()->size().height()){
        top = size.height() - pixmap()->size().height();
    }

    topleft.setX(left);
    topleft.setY(top);


    xScaled = (x + left) * pixmap()->size().width() / size.width();
    yScaled = (y + top) * pixmap()->size().height() / size.height();

    QPixmap aux2 = aux.copy(QRect(topleft, pixmap()->size()));

    this->setPixmap(aux2);


}
QImage MyLabel::Mat2QImage(const cv::Mat &src)
{
    cv::Mat temp; // make the same cv::Mat
    cvtColor(src, temp,CV_BGR2RGB); // cvtColor Makes a copt, that what i need
    QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    dest.bits(); // enforce deep copy, see documentation
    // of QImage::QImage ( const uchar * data, int width, int height, Format format )
    return dest;
}

void MyLabel::setImage(cv::Mat &image)
{
    QImage img= this->Mat2QImage(image);
    this->setPixmap(QPixmap::fromImage(img));
}

void MyLabel::setImage(QPixmap image)
{
    setPixmap(image);
  //  scaleImage(m_scaleFactor);
}

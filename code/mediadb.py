#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

# Basic test of using a mysql db to specify the inputs rather than just traversing a directory. 
# Note that this is mysql-specific - if you need completely generic code, head to:
# http://docs.sqlalchemy.org/en/latest/orm/tutorial.html
# to see how to make your code completely db-agnostic


__author__ = 'Paul Miller <paulymiller@gmail.com>'
__status__ = 'Prototype'
__application__ = 'soma'
__version__ = '0.0.1'
__date__ = '01-12-2016'
__copyright__ = 'Copyright 2015, Soma'

import os
import sys, getopt
import glob

#import pymysql

from sqlalchemy import create_engine
from sqlalchemy import ForeignKey, Column, select, and_, text
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import relationship, backref, column_property
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# initialize mapper Base
Base = declarative_base()


# class Image(Base):
#    __tablename__ = 'images'
#
#    id = Column(Integer, primary_key=True)
#
#    name = Column(String)
#    path = Column(String)
#    width = Column(Integer)
#    height = Column(Integer)
#
#    @classmethod
#    def _get_session(obj):
#        return Session.object_session(obj)
#
#    def _get_session(self):
#        return Session.object_session(self)

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    path = Column(String(200))
    width = Column(Integer)
    height = Column(Integer)

    def __repr__(self):
       return "<User(name='%s', path='%s', width='%d', height='%d')>" % (
                            self.name, self.path, self.width, self.height)

    # init is not really a constructor; it is literally an initialiser of instances
    def __init__(self, name="", path="", width=1, height=1):
        self.name = name;
        self.path = path
        self.width = width
        self.height = height


# Note: mysql requires you to specify a max length for varchar fields (i.e. the Strings below)
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    fullname = Column(String(200))
    password = Column(String(50))

    def __repr__(self):
       return "<User(name='%s', fullname='%s', password='%s')>" % (
                            self.name, self.fullname, self.password)

def main(argv):

    if len(argv) < 1:
        print("Usage: mediadb filename_pattern")
        print("E.g. mediadb /home/paul/Images/\*.jpg (remember to escape wildcards and spaces!)")
        return


    fileList = glob.glob(argv[0])
    print "found ", len(fileList), " files"

    imagelist = []
    for f in fileList:
        i = Image(name=os.path.basename(f), path=os.path.dirname(f))
        imagelist.append(i)

    print "length of imagelist: ", len(imagelist)
    for image in imagelist:
        print image.name

    engine = create_engine("mysql+pymysql://soma:amarillo@localhost/Soma")
    conn = engine.connect()

    Session = sessionmaker()
    Session.configure(bind=engine, autoflush=False)
    session = Session()

    # Slightly mysterious method that creates tables if they do not exist
    # Not clear yet how it knows which tables to create

    Base.metadata.create_all(engine)
#    session.add_all(imagelist)

    # (probably) inefficient method for doing a conditional insertion
    for img in imagelist:
        if not session.query(Image).filter_by(name=img.name).first():
            session.add(img)

    print "User.__table__: ", User.__table__

    user1 = User(name='Pug', fullname='Puggy Pearson', password='passwd')
    print(user1.__repr__())
    session.add(user1)
    print "Dirty? ", session.dirty
    tempUser = session.query(User).filter_by(name='Pug').first()
    print "tempUser :", tempUser
    print "user1: ", user1
    user2 = User(name='Doyle', fullname='Texas Dolly', password='passwd')
#    session.add(user2)

    print "Dirty? ", session.dirty
    print "session.new? ", session.new

    session.commit()
    print user1.id
    fake_user = User(name='Dummy', fullname='Invalid', password='12345')
    session.add(fake_user)
    print user1.name
    user1.name = 'Eddy'
    print "user1 name: ", user1.name
    print "Dirty? ", session.dirty
    print "session.new? ", session.new
    print "fake_user in session? ", fake_user in session
    session.commit()
    # Cannot rollback after committing!

    for instance in session.query(User).filter(User.name.in_(['Pug', 'Eddy', 'Dummy'])).all():
        print instance.name, instance.fullname
    # 'filter' is just a more powerful version of 'filter_by'

    tempUser = session.query(User).filter_by(name='Eddy').first()
    print "tempuser: ", tempUser
    session.rollback()
    print "user1 name: ", user1.name
    print "fake_user in session? ", fake_user in session

    print "** Let's try a simple select command"

    # for instance in session.query(User). order_by(User.id):
    #     print instance, instance.name, instance.fullname
    #
    # print "** And now another select"
    #
    # for name, fullname in session.query(User.name, User.fullname):
    #     print name, fullname
    #
    # # The 'name_label' string just gives a name for the element of row that we query later.
    # for row in session.query(User.name.label('name_label')).all():
    #     print row
    #
    # for u in session.query(User).order_by(User.id)[1:4]:
    #     print u

    for name in session.query(User.name).\
        filter(User.fullname.in_(['Puggy Pearson', 'Invalid'])):
        print name

    stmt = text("SELECT name, id, fullname, password "
                "FROM users where name=:name")
    stmt = stmt.columns(User.id, User.name, User.fullname, User.password)
    res = session.query(User).from_statement(stmt).params(name='Eddy').all()

    print "stmt: ", stmt
    print "res: ", res


if (__name__ == '__main__'):
    main(sys.argv[1:])
